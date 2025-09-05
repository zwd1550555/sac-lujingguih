# -*- coding: utf-8 -*-
"""
序列化训练器 - 实现真正的BPTT训练
充分利用液态神经网络的时序记忆能力

主要功能:
- 序列化更新: 使用BPTT进行序列化训练
- 时序依赖学习: 学习跨越多个时间步的依赖关系
- 隐藏状态管理: 正确处理LNNs的隐藏状态传递
- 梯度裁剪: 防止梯度爆炸问题

技术特性:
- BPTT训练: 随时间反向传播，学习长期依赖
- 序列采样: 从经验池中采样完整序列
- 梯度管理: 智能梯度裁剪和累积
- 内存优化: 高效的序列处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from agent_lnn import SequenceReplayBuffer
from agent_advanced import AdvancedLiquidSACAgent


class BPTTTrainer:
    """
    随时间反向传播训练器
    """
    
    def __init__(self, agent: AdvancedLiquidSACAgent, 
                 sequence_length: int = 10,
                 gradient_clip_norm: float = 1.0,
                 gamma_sequence: float = 0.99):
        self.agent = agent
        self.sequence_length = sequence_length
        self.gradient_clip_norm = gradient_clip_norm
        self.gamma_sequence = gamma_sequence
        
    def compute_sequence_returns(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        计算序列的折扣回报
        
        Args:
            rewards: 奖励序列 (sequence_length,)
            dones: 终止状态序列 (sequence_length,)
            
        Returns:
            returns: 折扣回报序列 (sequence_length,)
        """
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        # 从后往前计算折扣回报
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma_sequence * running_return
            returns[t] = running_return
            
        return returns
    
    def train_actor_sequence(self, sequences: List[Dict]) -> Dict[str, float]:
        """
        使用BPTT训练Actor网络
        
        Args:
            sequences: 序列数据列表
            
        Returns:
            losses: 损失信息
        """
        if not sequences:
            return {}
        
        batch_size = len(sequences)
        sequence_length = self.sequence_length
        
        # 准备序列数据
        states = torch.zeros(batch_size, sequence_length, self.agent.actor.state_dim, 
                           device=self.agent.device)
        actions = torch.zeros(batch_size, sequence_length, self.agent.actor.action_dim, 
                            device=self.agent.device)
        rewards = torch.zeros(batch_size, sequence_length, device=self.agent.device)
        dones = torch.zeros(batch_size, sequence_length, device=self.agent.device)
        returns = torch.zeros(batch_size, sequence_length, device=self.agent.device)
        
        # 填充序列数据
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq['states']), sequence_length)
            states[i, :seq_len] = torch.FloatTensor(seq['states'][:seq_len])
            actions[i, :seq_len] = torch.FloatTensor(seq['actions'][:seq_len])
            rewards[i, :seq_len] = torch.FloatTensor(seq['rewards'][:seq_len])
            dones[i, :seq_len] = torch.FloatTensor(seq['dones'][:seq_len])
            
            # 计算折扣回报
            returns[i, :seq_len] = self.compute_sequence_returns(
                rewards[i, :seq_len], dones[i, :seq_len]
            )
        
        # 初始化隐藏状态
        actor_hidden = None
        critic_hidden1 = None
        critic_hidden2 = None
        
        # 前向传播 - 计算整个序列的策略和Q值
        actor_losses = []
        critic_losses = []
        
        for t in range(sequence_length):
            # 当前时间步的状态和动作
            state_t = states[:, t, :]  # (batch_size, state_dim)
            action_t = actions[:, t, :]  # (batch_size, action_dim)
            return_t = returns[:, t]  # (batch_size,)
            
            # Actor前向传播
            mean, log_std, actor_hidden = self.agent.actor(state_t, actor_hidden)
            std = log_std.exp()
            
            # 计算策略损失
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(action_t).sum(dim=-1)
            
            # 使用当前Q值作为基线
            with torch.no_grad():
                q1, q2, _, _ = self.agent.critic(state_t, action_t, critic_hidden1, critic_hidden2)
                q_value = torch.min(q1, q2).squeeze()
            
            # 策略梯度损失
            actor_loss = -(log_prob * return_t).mean()
            actor_losses.append(actor_loss)
            
            # Critic前向传播
            q1, q2, critic_hidden1, critic_hidden2 = self.agent.critic(state_t, action_t, critic_hidden1, critic_hidden2)
            
            # Critic损失
            critic_loss = F.mse_loss(q1.squeeze(), return_t) + F.mse_loss(q2.squeeze(), return_t)
            critic_losses.append(critic_loss)
        
        # 计算总损失
        total_actor_loss = torch.stack(actor_losses).mean()
        total_critic_loss = torch.stack(critic_losses).mean()
        
        # 反向传播
        self.agent.actor_opt.zero_grad()
        total_actor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.agent.actor.parameters(), self.gradient_clip_norm)
        self.agent.actor_opt.step()
        
        self.agent.critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.gradient_clip_norm)
        self.agent.critic_opt.step()
        
        return {
            'actor_loss': total_actor_loss.item(),
            'critic_loss': total_critic_loss.item(),
            'sequence_length': sequence_length,
            'batch_size': batch_size
        }
    
    def train_critic_sequence(self, sequences: List[Dict]) -> Dict[str, float]:
        """
        使用BPTT训练Critic网络
        
        Args:
            sequences: 序列数据列表
            
        Returns:
            losses: 损失信息
        """
        if not sequences:
            return {}
        
        batch_size = len(sequences)
        sequence_length = self.sequence_length
        
        # 准备序列数据
        states = torch.zeros(batch_size, sequence_length, self.agent.actor.state_dim, 
                           device=self.agent.device)
        actions = torch.zeros(batch_size, sequence_length, self.agent.actor.action_dim, 
                            device=self.agent.device)
        rewards = torch.zeros(batch_size, sequence_length, device=self.agent.device)
        dones = torch.zeros(batch_size, sequence_length, device=self.agent.device)
        next_states = torch.zeros(batch_size, sequence_length, self.agent.actor.state_dim, 
                                device=self.agent.device)
        
        # 填充序列数据
        for i, seq in enumerate(sequences):
            seq_len = min(len(seq['states']), sequence_length)
            states[i, :seq_len] = torch.FloatTensor(seq['states'][:seq_len])
            actions[i, :seq_len] = torch.FloatTensor(seq['actions'][:seq_len])
            rewards[i, :seq_len] = torch.FloatTensor(seq['rewards'][:seq_len])
            dones[i, :seq_len] = torch.FloatTensor(seq['dones'][:seq_len])
            next_states[i, :seq_len] = torch.FloatTensor(seq['next_states'][:seq_len])
        
        # 初始化隐藏状态
        critic_hidden1 = None
        critic_hidden2 = None
        target_critic_hidden1 = None
        target_critic_hidden2 = None
        
        # 计算目标Q值
        with torch.no_grad():
            target_q_values = []
            for t in range(sequence_length):
                if t < sequence_length - 1:
                    next_state_t = next_states[:, t, :]
                    # 使用目标网络计算下一状态的Q值
                    target_q1, target_q2, target_critic_hidden1, target_critic_hidden2 = \
                        self.agent.target_critic(next_state_t, actions[:, t+1, :], 
                                               target_critic_hidden1, target_critic_hidden2)
                    target_q = torch.min(target_q1, target_q2)
                    target_q_values.append(target_q.squeeze())
                else:
                    target_q_values.append(torch.zeros(batch_size, device=self.agent.device))
        
        # 前向传播和损失计算
        critic_losses = []
        
        for t in range(sequence_length):
            state_t = states[:, t, :]
            action_t = actions[:, t, :]
            reward_t = rewards[:, t]
            done_t = dones[:, t]
            target_q_t = target_q_values[t]
            
            # 计算当前Q值
            q1, q2, critic_hidden1, critic_hidden2 = self.agent.critic(state_t, action_t, 
                                                                     critic_hidden1, critic_hidden2)
            
            # 计算目标Q值
            target_q = reward_t + (1 - done_t) * self.agent.gamma * target_q_t
            
            # Critic损失
            critic_loss = F.mse_loss(q1.squeeze(), target_q) + F.mse_loss(q2.squeeze(), target_q)
            critic_losses.append(critic_loss)
        
        # 计算总损失
        total_critic_loss = torch.stack(critic_losses).mean()
        
        # 反向传播
        self.agent.critic_opt.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.critic.parameters(), self.gradient_clip_norm)
        self.agent.critic_opt.step()
        
        return {
            'critic_loss': total_critic_loss.item(),
            'sequence_length': sequence_length,
            'batch_size': batch_size
        }


class SequenceBasedSACAgent(AdvancedLiquidSACAgent):
    """
    基于序列的SAC智能体
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 replay_buffer_capacity: int = 1_000_000, actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, target_update_interval: int = 1,
                 sequence_length: int = 10, num_candidates: int = 5,
                 use_intervention_decision: bool = True,
                 use_sequence_training: bool = True):
        
        super().__init__(state_dim, action_dim, hidden_dim, replay_buffer_capacity,
                        actor_lr, critic_lr, gamma, tau, alpha, target_update_interval,
                        sequence_length, num_candidates, use_intervention_decision)
        
        self.use_sequence_training = use_sequence_training
        self.bptt_trainer = BPTTTrainer(self, sequence_length)
        
    def update(self, batch_size: int):
        """
        更新智能体 - 支持序列化训练
        
        Args:
            batch_size: 批次大小
            
        Returns:
            metrics: 训练指标
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        if self.use_sequence_training:
            # 使用序列化训练
            sequences = self.replay_buffer.sample_sequence(batch_size)
            if sequences is not None:
                # 训练Actor
                actor_metrics = self.bptt_trainer.train_actor_sequence(sequences)
                
                # 训练Critic
                critic_metrics = self.bptt_trainer.train_critic_sequence(sequences)
                
                # 更新目标网络
                self._updates += 1
                if self._updates % self.target_update_interval == 0:
                    self._soft_update(self.target_critic, self.critic, self.tau)
                
                return {
                    'actor_loss': actor_metrics.get('actor_loss', 0),
                    'critic_loss': critic_metrics.get('critic_loss', 0),
                    'sequence_length': actor_metrics.get('sequence_length', 0),
                    'batch_size': actor_metrics.get('batch_size', 0),
                    'training_mode': 'sequence'
                }
        
        # 回退到标准训练
        return super().update(batch_size)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """
        软更新目标网络
        
        Args:
            target: 目标网络
            source: 源网络
            tau: 更新系数
        """
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class SequenceAnalyzer:
    """
    序列分析器 - 分析序列训练效果
    """
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        
    def analyze_sequence_patterns(self, sequences: List[Dict]) -> Dict[str, float]:
        """
        分析序列模式
        
        Args:
            sequences: 序列数据列表
            
        Returns:
            analysis: 分析结果
        """
        if not sequences:
            return {}
        
        # 计算序列统计信息
        total_rewards = []
        sequence_lengths = []
        success_rates = []
        
        for seq in sequences:
            total_reward = sum(seq['rewards'])
            seq_len = len(seq['states'])
            success = seq['dones'][-1] if seq['dones'] else False
            
            total_rewards.append(total_reward)
            sequence_lengths.append(seq_len)
            success_rates.append(1.0 if success else 0.0)
        
        return {
            'avg_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'avg_sequence_length': np.mean(sequence_lengths),
            'success_rate': np.mean(success_rates),
            'num_sequences': len(sequences)
        }
    
    def detect_sequence_anomalies(self, sequences: List[Dict]) -> List[int]:
        """
        检测序列异常
        
        Args:
            sequences: 序列数据列表
            
        Returns:
            anomaly_indices: 异常序列的索引
        """
        if not sequences:
            return []
        
        anomaly_indices = []
        
        for i, seq in enumerate(sequences):
            # 检测异常：奖励过大或过小
            total_reward = sum(seq['rewards'])
            if abs(total_reward) > 1000:  # 异常阈值
                anomaly_indices.append(i)
            
            # 检测异常：序列长度异常
            seq_len = len(seq['states'])
            if seq_len < 2 or seq_len > self.sequence_length * 2:
                anomaly_indices.append(i)
        
        return anomaly_indices
