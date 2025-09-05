# -*- coding: utf-8 -*-
"""
世界模型 - 用于因果强化学习的环境预测
实现反事实推理和干预预测

主要功能:
- 环境状态转移预测: 预测下一状态、奖励和终止条件
- 反事实推理: 计算"如果采取其他动作会怎样"的结果
- 干预预测: 预测特定干预动作的长期结果
- 因果信誉分配: 精确评估动作的真实贡献

技术特性:
- 神经网络建模: 使用深度神经网络学习环境动态
- 序列预测: 支持多步前向预测
- 因果建模: 理解环境中的因果关系
- 在线学习: 支持在线更新和适应

核心组件:
- WorldModel: 环境动态预测模型
- CausalReasoner: 因果推理器
- WorldModelTrainer: 世界模型训练器

使用方法:
world_model = WorldModel(state_dim=22, action_dim=2, hidden_dim=256)
causal_reasoner = CausalReasoner(world_model, device)
trainer = WorldModelTrainer(world_model, lr=1e-3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class WorldModel(nn.Module):
    """世界模型：预测环境状态转移"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 动作编码器
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 状态转移预测器
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # 奖励预测器
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 终止状态预测器
        self.done_predictor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        前向传播：预测下一状态、奖励和终止状态
        
        Args:
            state: 当前状态 (batch_size, state_dim)
            action: 当前动作 (batch_size, action_dim)
            
        Returns:
            next_state: 预测的下一状态
            reward: 预测的奖励
            done: 预测的终止状态
        """
        # 编码状态和动作
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        
        # 拼接状态和动作编码
        combined = torch.cat([state_encoded, action_encoded], dim=1)
        
        # 预测下一状态、奖励和终止状态
        next_state = self.transition_predictor(combined)
        reward = self.reward_predictor(combined)
        done = self.done_predictor(combined)
        
        return next_state, reward, done.squeeze()
    
    def predict_sequence(self, initial_state: torch.Tensor, actions: torch.Tensor):
        """
        预测动作序列的结果
        
        Args:
            initial_state: 初始状态 (batch_size, state_dim)
            actions: 动作序列 (batch_size, sequence_length, action_dim)
            
        Returns:
            states: 预测的状态序列
            rewards: 预测的奖励序列
            dones: 预测的终止状态序列
        """
        batch_size, seq_len, _ = actions.shape
        states = [initial_state]
        rewards = []
        dones = []
        
        current_state = initial_state
        for t in range(seq_len):
            action = actions[:, t, :]
            next_state, reward, done = self.forward(current_state, action)
            
            states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            
            # 如果预测终止，则停止
            if done.any():
                break
                
            current_state = next_state
        
        return torch.stack(states[1:], dim=1), torch.stack(rewards, dim=1), torch.stack(dones, dim=1)


class CausalReasoner:
    """因果推理器：实现反事实推理和干预预测"""
    
    def __init__(self, world_model: WorldModel, device: torch.device):
        self.world_model = world_model
        self.device = device
    
    def counterfactual_reward(self, state: torch.Tensor, actual_action: torch.Tensor, 
                            default_action: torch.Tensor, actual_reward: float) -> float:
        """
        计算反事实奖励：如果采取默认动作会得到什么奖励
        
        Args:
            state: 当前状态
            actual_action: 实际采取的动作
            default_action: 默认动作
            actual_reward: 实际获得的奖励
            
        Returns:
            counterfactual_reward: 反事实奖励
        """
        with torch.no_grad():
            # 预测默认动作的结果
            _, predicted_reward, _ = self.world_model(state, default_action)
            counterfactual_reward = predicted_reward.item()
            
            # 计算因果贡献：实际奖励 - 反事实奖励
            causal_contribution = actual_reward - counterfactual_reward
            
            return causal_contribution
    
    def intervention_prediction(self, state: torch.Tensor, intervention_action: torch.Tensor, 
                              horizon: int = 5) -> dict:
        """
        干预预测：预测采取特定干预动作的长期结果
        
        Args:
            state: 当前状态
            intervention_action: 干预动作
            horizon: 预测时间范围
            
        Returns:
            prediction: 包含预测结果的字典
        """
        with torch.no_grad():
            current_state = state
            states = [current_state]
            rewards = []
            dones = []
            
            for t in range(horizon):
                # 预测下一状态
                next_state, reward, done = self.world_model(current_state, intervention_action)
                
                states.append(next_state)
                rewards.append(reward.item())
                dones.append(done.item())
                
                if done.item():
                    break
                    
                current_state = next_state
            
            return {
                'states': torch.stack(states[1:], dim=0),
                'rewards': rewards,
                'dones': dones,
                'total_reward': sum(rewards),
                'safety_score': self._calculate_safety_score(states[1:])
            }
    
    def _calculate_safety_score(self, states: list) -> float:
        """
        计算安全分数：基于预测状态的安全性评估
        
        Args:
            states: 预测的状态序列
            
        Returns:
            safety_score: 安全分数 (0-1，1表示最安全)
        """
        if not states:
            return 0.0
        
        # 简化的安全评估：检查速度是否在安全范围内
        safety_scores = []
        for state in states:
            # 假设状态的前几个维度包含速度信息
            speed = abs(state[0].item())  # 归一化速度
            if speed < 0.8:  # 安全速度范围
                safety_scores.append(1.0)
            else:
                safety_scores.append(max(0.0, 1.0 - (speed - 0.8) * 2))
        
        return np.mean(safety_scores) if safety_scores else 0.0
    
    def causal_credit_assignment(self, state: torch.Tensor, action: torch.Tensor, 
                               reward: float, next_state: torch.Tensor) -> dict:
        """
        因果信誉分配：分析动作对结果的因果贡献
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            
        Returns:
            credit_info: 因果信誉信息
        """
        with torch.no_grad():
            # 定义几个基准动作
            baseline_actions = [
                torch.tensor([0.0, 0.0], device=self.device),  # 停止
                torch.tensor([0.3, 0.3], device=self.device),  # 直行
                torch.tensor([0.2, 0.4], device=self.device),  # 左转
                torch.tensor([0.4, 0.2], device=self.device),  # 右转
            ]
            
            # 计算每个基准动作的预测结果
            baseline_results = []
            for baseline_action in baseline_actions:
                pred_next_state, pred_reward, pred_done = self.world_model(state, baseline_action)
                baseline_results.append({
                    'action': baseline_action,
                    'reward': pred_reward.item(),
                    'done': pred_done.item()
                })
            
            # 找到最佳基准动作
            best_baseline = max(baseline_results, key=lambda x: x['reward'])
            
            # 计算因果贡献
            causal_contribution = reward - best_baseline['reward']
            
            return {
                'actual_reward': reward,
                'best_baseline_reward': best_baseline['reward'],
                'causal_contribution': causal_contribution,
                'baseline_results': baseline_results
            }


class WorldModelTrainer:
    """世界模型训练器"""
    
    def __init__(self, world_model: WorldModel, lr: float = 1e-3):
        self.world_model = world_model
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                   next_states: torch.Tensor, rewards: torch.Tensor, 
                   dones: torch.Tensor) -> dict:
        """
        训练世界模型一步
        
        Args:
            states: 当前状态
            actions: 动作
            next_states: 下一状态
            rewards: 奖励
            dones: 终止状态
            
        Returns:
            losses: 损失信息
        """
        self.optimizer.zero_grad()
        
        # 前向传播
        pred_next_states, pred_rewards, pred_dones = self.world_model(states, actions)
        
        # 计算损失
        state_loss = self.mse_loss(pred_next_states, next_states)
        reward_loss = self.mse_loss(pred_rewards.squeeze(), rewards)
        done_loss = self.bce_loss(pred_dones, dones.float())
        
        total_loss = state_loss + reward_loss + done_loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'state_loss': state_loss.item(),
            'reward_loss': reward_loss.item(),
            'done_loss': done_loss.item()
        }
    
    def train_from_buffer(self, replay_buffer, batch_size: int = 256, epochs: int = 1):
        """
        从经验回放缓冲区训练世界模型
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            epochs: 训练轮数
            
        Returns:
            avg_losses: 平均损失
        """
        if len(replay_buffer) < batch_size:
            return None
        
        total_losses = {'total_loss': 0, 'state_loss': 0, 'reward_loss': 0, 'done_loss': 0}
        num_batches = 0
        
        for epoch in range(epochs):
            # 随机采样批次
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # 转换为张量
            states = torch.FloatTensor(states).to(self.world_model.device)
            actions = torch.FloatTensor(actions).to(self.world_model.device)
            next_states = torch.FloatTensor(next_states).to(self.world_model.device)
            rewards = torch.FloatTensor(rewards).to(self.world_model.device)
            dones = torch.FloatTensor(dones).to(self.world_model.device)
            
            # 训练一步
            losses = self.train_step(states, actions, next_states, rewards, dones)
            
            # 累积损失
            for key, value in losses.items():
                total_losses[key] += value
            num_batches += 1
        
        # 计算平均损失
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        return avg_losses
