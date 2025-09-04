# -*- coding: utf-8 -*-
"""
Soft Actor-Critic (SAC) agent implementation for onsite_mine training.
Includes: ReplayBuffer, Actor, Critic (double Q), SACAgent with adaptive alpha.
"""

from collections import deque
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int = None, action_dim: int = None, dtype=np.float32):
        self.capacity = int(capacity)
        self.size = 0
        self.ptr = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dtype = dtype
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

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

    def push(self, state, action, reward, next_state, done):
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
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
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


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), self.log_std_min, self.log_std_max)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        # Q1
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        # Q2
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


class SACAgent:
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
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = max(1, int(target_update_interval))
        self._updates = 0

        # Entropy temperature
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=actor_lr)
        self._alpha_cached = None
        self.alpha = torch.tensor(alpha, device=self.device)
        self.alpha_learn = True  # 可由外部关闭并改为调度

    @torch.no_grad()
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, log_std = self.actor(state_t)
        std = log_std.exp()
        dist = Normal(mean, std)
        if evaluate:
            action = torch.tanh(mean)
        else:
            z = dist.rsample()
            action = torch.tanh(z)
        return action.squeeze(0).cpu().numpy()

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    def update(self, batch_size: int):
        if len(self.replay_buffer) < batch_size:
            return None

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mean_next, log_std_next = self.actor(next_state)
            std_next = log_std_next.exp()
            dist_next = Normal(mean_next, std_next)
            z_next = dist_next.rsample()
            action_next = torch.tanh(z_next)
            log_prob_next = dist_next.log_prob(z_next) - torch.log(1 - action_next.pow(2) + 1e-6)
            log_prob_next = log_prob_next.sum(dim=1, keepdim=True)

            q1_t, q2_t = self.target_critic(next_state, action_next)
            # 缓存 alpha.exp() 避免重复求 exp
            if self._alpha_cached is None:
                self._alpha_cached = self.alpha.exp()
            q_t = torch.min(q1_t, q2_t) - self._alpha_cached * log_prob_next
            target_q = reward + (1 - done) * self.gamma * q_t

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        mean, log_std = self.actor(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        pi_action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - pi_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        q1_pi, q2_pi = self.critic(state, pi_action)
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
            # 未学习 alpha 时记录为 0 方便日志
            alpha_loss = torch.zeros(1, device=self.device)
        # 更新缓存
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


