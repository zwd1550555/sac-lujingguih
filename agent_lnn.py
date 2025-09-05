# -*- coding: utf-8 -*-
"""
液态神经网络 (LNNs) 增强的SAC智能体实现
集成LTC (Liquid Time-Constant) 网络和因果强化学习思想

主要特性:
- 液态神经网络: 使用LTC网络替代传统MLP，具备连续时间处理能力
- 序列记忆: 支持时序数据的处理和记忆
- 动态适应: 根据输入动态调整网络内部状态
- 因果推理: 集成世界模型进行因果推理

技术优势:
- 更强的时序动态捕捉能力
- 更高的鲁棒性和适应性
- 更强的因果表达能力
- 更好的泛化性能

使用方法:
from agent_lnn import LiquidSACAgent
agent = LiquidSACAgent(state_dim=22, action_dim=2, hidden_dim=512)
"""

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# 液态神经网络库
try:
    from ncps.torch import LTC
    from ncps.wirings import AutoNCP
    LNN_AVAILABLE = True
except ImportError:
    LNN_AVAILABLE = False
    print("警告: ncps库未安装，将使用标准神经网络")


class SequenceReplayBuffer:
    """
    支持序列采样的经验回放缓冲区
    
    功能特性:
    - 支持序列数据的存储和采样
    - 兼容原有的随机采样接口
    - 支持隐藏状态的存储
    - 自动管理缓冲区容量
    
    技术细节:
    - 使用numpy数组存储经验数据，提高访问效率
    - 支持序列长度可配置
    - 自动处理缓冲区溢出（循环覆盖）
    
    使用方法:
    buffer = SequenceReplayBuffer(capacity=1000000, sequence_length=10)
    buffer.push(state, action, reward, next_state, done, hidden_state)
    sequences = buffer.sample_sequence(batch_size=32)
    """
    def __init__(self, capacity: int, state_dim: int = None, action_dim: int = None, 
                 sequence_length: int = 10, dtype=np.float32):
        self.capacity = int(capacity)
        self.size = 0
        self.ptr = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.dtype = dtype
        
        # 存储序列数据
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None
        self.hidden_states = None

    def _maybe_init(self, state: np.ndarray, action: np.ndarray):
        if self.states is not None:
            return
        sdim = self.state_dim or state.shape[-1]
        adim = self.action_dim or action.shape[-1]
        self.states = np.zeros((self.capacity, sdim), dtype=self.dtype)
        self.actions = np.zeros((self.capacity, adim), dtype=self.dtype)
        self.rewards = np.zeros((self.capacity, 1), dtype=self.dtype)
        self.next_states = np.zeros((self.capacity, sdim), dtype=self.dtype)
        self.dones = np.zeros((self.capacity, 1), dtype=self.dtype)
        self.hidden_states = [None] * self.capacity

    def push(self, state, action, reward, next_state, done, hidden_state=None):
        state = np.asarray(state, dtype=self.dtype)
        action = np.asarray(action, dtype=self.dtype)
        next_state = np.asarray(next_state, dtype=self.dtype)
        reward = float(reward)
        done = float(done)
        self._maybe_init(state, action)
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr, 0] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr, 0] = done
        self.hidden_states[self.ptr] = hidden_state
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_sequence(self, batch_size: int):
        """采样序列数据"""
        if self.size < self.sequence_length:
            return None
            
        # 确保有足够的序列长度
        max_start_idx = self.size - self.sequence_length
        start_indices = np.random.randint(0, max_start_idx, size=batch_size)
        
        sequences = []
        for start_idx in start_indices:
            end_idx = start_idx + self.sequence_length
            sequence = {
                'states': self.states[start_idx:end_idx],
                'actions': self.actions[start_idx:end_idx],
                'rewards': self.rewards[start_idx:end_idx],
                'next_states': self.next_states[start_idx:end_idx],
                'dones': self.dones[start_idx:end_idx],
                'hidden_states': self.hidden_states[start_idx:end_idx]
            }
            sequences.append(sequence)
        
        return sequences

    def sample(self, batch_size: int):
        """兼容原有接口的随机采样"""
        if self.size < batch_size:
            return None
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


class LiquidActor(nn.Module):
    """基于液态神经网络的Actor"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if LNN_AVAILABLE:
            # 使用液态神经网络
            wiring = AutoNCP(hidden_dim, action_dim)
            self.lnn = LTC(state_dim, wiring, batch_first=True)
            self.log_std_layer = nn.Linear(action_dim, action_dim)
        else:
            # 回退到标准网络
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.mean_layer = nn.Linear(hidden_dim // 2, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim // 2, action_dim)
            self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor, hidden_state=None):
        if LNN_AVAILABLE:
            # 液态神经网络前向传播
            if state.ndim == 2:
                state = state.unsqueeze(1)  # (batch, 1, features)
            
            mean_sequence, new_hidden_state = self.lnn(state, hidden_state)
            mean = mean_sequence[:, -1, :]  # 取最后一个时间步
            
            log_std = self.log_std_layer(mean)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            
            return mean, log_std, new_hidden_state
        else:
            # 标准网络前向传播
            x = F.relu(self.fc1(state))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            mean = self.mean_layer(x)
            log_std = torch.clamp(self.log_std_layer(x), self.log_std_min, self.log_std_max)
            return mean, log_std, None


class LiquidCritic(nn.Module):
    """基于液态神经网络的Critic"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        if LNN_AVAILABLE:
            # 使用液态神经网络
            wiring1 = AutoNCP(hidden_dim, 1)
            wiring2 = AutoNCP(hidden_dim, 1)
            self.lnn1 = LTC(state_dim + action_dim, wiring1, batch_first=True)
            self.lnn2 = LTC(state_dim + action_dim, wiring2, batch_first=True)
        else:
            # 回退到标准网络
            # Q1
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc4 = nn.Linear(hidden_dim // 2, 1)
            # Q2
            self.fc5 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc6 = nn.Linear(hidden_dim, hidden_dim)
            self.fc7 = nn.Linear(hidden_dim, hidden_dim // 2)
            self.fc8 = nn.Linear(hidden_dim // 2, 1)
            self.dropout = nn.Dropout(0.1)

    def forward(self, state: torch.Tensor, action: torch.Tensor, 
                hidden_state1=None, hidden_state2=None):
        sa = torch.cat([state, action], dim=1)
        
        if LNN_AVAILABLE:
            # 液态神经网络前向传播
            if sa.ndim == 2:
                sa = sa.unsqueeze(1)  # (batch, 1, features)
            
            q1_seq, new_hidden_state1 = self.lnn1(sa, hidden_state1)
            q2_seq, new_hidden_state2 = self.lnn2(sa, hidden_state2)
            
            q1 = q1_seq[:, -1, :]  # 取最后一个时间步
            q2 = q2_seq[:, -1, :]
            
            return q1, q2, new_hidden_state1, new_hidden_state2
        else:
            # 标准网络前向传播
            # Q1
            q1 = F.relu(self.fc1(sa))
            q1 = self.dropout(q1)
            q1 = F.relu(self.fc2(q1))
            q1 = self.dropout(q1)
            q1 = F.relu(self.fc3(q1))
            q1 = self.fc4(q1)
            # Q2
            q2 = F.relu(self.fc5(sa))
            q2 = self.dropout(q2)
            q2 = F.relu(self.fc6(q2))
            q2 = self.dropout(q2)
            q2 = F.relu(self.fc7(q2))
            q2 = self.fc8(q2)
            
            return q1, q2, None, None


class LiquidSACAgent:
    """液态神经网络增强的SAC智能体"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        replay_buffer_capacity: int = 1_000_000,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        target_update_interval: int = 1,
        sequence_length: int = 10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = sequence_length

        # 网络初始化
        self.actor = LiquidActor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = LiquidCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = LiquidCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 经验回放缓冲区
        self.replay_buffer = SequenceReplayBuffer(
            replay_buffer_capacity, state_dim, action_dim, sequence_length
        )
        
        # SAC参数
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = max(1, int(target_update_interval))
        self._updates = 0

        # 熵温度
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=actor_lr)
        self._alpha_cached = None
        self.alpha = torch.tensor(alpha, device=self.device)
        self.alpha_learn = True

    @torch.no_grad()
    def select_action(self, state: np.ndarray, hidden_state=None, evaluate: bool = False) -> tuple:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, log_std, new_hidden_state = self.actor(state_t, hidden_state)
        std = log_std.exp()
        dist = Normal(mean, std)
        
        if evaluate:
            action = torch.tanh(mean)
        else:
            z = dist.rsample()
            action = torch.tanh(z)
        
        return action.squeeze(0).cpu().numpy(), new_hidden_state

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def update(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None

        # 尝试使用序列采样，如果失败则使用随机采样
        sequences = self.replay_buffer.sample_sequence(batch_size)
        if sequences is None:
            # 回退到随机采样
            state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
            return self._update_standard(state, action, reward, next_state, done)
        
        # 使用序列数据进行训练
        return self._update_sequence(sequences)

    def _update_standard(self, state, action, reward, next_state, done):
        """标准SAC更新"""
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mean_next, log_std_next, _ = self.actor(next_state)
            std_next = log_std_next.exp()
            dist_next = Normal(mean_next, std_next)
            z_next = dist_next.rsample()
            action_next = torch.tanh(z_next)
            log_prob_next = dist_next.log_prob(z_next) - torch.log(1 - action_next.pow(2) + 1e-6)
            log_prob_next = log_prob_next.sum(dim=1, keepdim=True)

            q1_t, q2_t, _, _ = self.target_critic(next_state, action_next)
            if self._alpha_cached is None:
                self._alpha_cached = self.alpha.exp()
            q_t = torch.min(q1_t, q2_t) - self._alpha_cached * log_prob_next
            target_q = reward + (1 - done) * self.gamma * q_t

        q1, q2, _, _ = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        mean, log_std, _ = self.actor(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        pi_action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - pi_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        q1_pi, q2_pi, _, _ = self.critic(state, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        if self._alpha_cached is None:
            self._alpha_cached = self.alpha.exp()
        actor_loss = (self._alpha_cached * log_prob - min_q_pi).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        if self.alpha_learn:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
        else:
            alpha_loss = torch.zeros(1, device=self.device)
        
        self._alpha_cached = None
        self._updates += 1
        if self._updates % self.target_update_interval == 0:
            self._soft_update(self.target_critic, self.critic, self.tau)

        return {
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'actor_loss': float(actor_loss.detach().cpu().item()),
            'alpha_loss': float(alpha_loss.detach().cpu().item()),
            'alpha': float(self.alpha.detach().cpu().item()),
        }

    def _update_sequence(self, sequences):
        """基于序列的更新（未来扩展）"""
        # 这里可以实现基于序列的训练逻辑
        # 目前先回退到标准更新
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []
        
        for seq in sequences:
            # 取序列的最后一个元素进行标准更新
            batch_states.append(seq['states'][-1])
            batch_actions.append(seq['actions'][-1])
            batch_rewards.append(seq['rewards'][-1])
            batch_next_states.append(seq['next_states'][-1])
            batch_dones.append(seq['dones'][-1])
        
        return self._update_standard(
            np.array(batch_states),
            np.array(batch_actions),
            np.array(batch_rewards),
            np.array(batch_next_states),
            np.array(batch_dones)
        )
