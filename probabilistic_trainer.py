# -*- coding: utf-8 -*-
"""
概率世界模型训练器
实现基于负对数似然损失的训练方法

主要功能:
- 概率世界模型训练: 使用负对数似然损失训练概率预测模型
- 不确定性量化: 准确评估模型预测的不确定性
- 自适应训练: 根据不确定性调整训练策略
- 鲁棒性提升: 提高模型对噪声和不确定性的鲁棒性

技术特性:
- 负对数似然损失: 迫使模型准确评估预测不确定性
- 概率分布输出: 输出预测的均值和标准差
- 不确定性感知: 基于不确定性进行决策
- 在线适应: 支持在线更新和适应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from agent_advanced import ProbabilisticWorldModel


class ProbabilisticWorldModelTrainer:
    """
    概率世界模型训练器
    """
    
    def __init__(self, world_model: ProbabilisticWorldModel, lr: float = 1e-3,
                 kl_weight: float = 0.1, uncertainty_threshold: float = 0.5):
        self.world_model = world_model
        self.optimizer = torch.optim.Adam(world_model.parameters(), lr=lr)
        self.kl_weight = kl_weight  # KL散度权重
        self.uncertainty_threshold = uncertainty_threshold
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
    def negative_log_likelihood_loss(self, pred_mean: torch.Tensor, pred_logstd: torch.Tensor, 
                                   target: torch.Tensor) -> torch.Tensor:
        """
        计算负对数似然损失
        
        Args:
            pred_mean: 预测均值
            pred_logstd: 预测对数标准差
            target: 真实值
            
        Returns:
            nll_loss: 负对数似然损失
        """
        pred_std = pred_logstd.exp()
        
        # 计算负对数似然
        nll = 0.5 * ((target - pred_mean) / pred_std) ** 2 + pred_logstd + 0.5 * np.log(2 * np.pi)
        nll_loss = nll.mean()
        
        return nll_loss
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor, 
                   next_states: torch.Tensor, rewards: torch.Tensor, 
                   dones: torch.Tensor) -> Dict[str, float]:
        """
        训练概率世界模型一步
        
        Args:
            states: 当前状态
            actions: 动作
            next_states: 下一状态
            rewards: 奖励
            dones: 终止状态
            
        Returns:
            losses: 损失信息字典
        """
        self.optimizer.zero_grad()
        
        # 前向传播
        next_state_mean, next_state_logstd, reward_mean, reward_logstd, pred_dones = self.world_model(states, actions)
        
        # 计算损失
        # 状态转移损失（负对数似然）
        state_nll_loss = self.negative_log_likelihood_loss(next_state_mean, next_state_logstd, next_states)
        
        # 奖励预测损失（负对数似然）
        reward_nll_loss = self.negative_log_likelihood_loss(reward_mean, reward_logstd, rewards.unsqueeze(-1))
        
        # 终止状态损失（二元交叉熵）
        done_loss = self.bce_loss(pred_dones, dones.float())
        
        # 正则化项：防止标准差过大
        std_regularization = torch.mean(next_state_logstd.exp() + reward_logstd.exp())
        
        # 总损失
        total_loss = (state_nll_loss + reward_nll_loss + done_loss + 
                     self.kl_weight * std_regularization)
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'state_nll_loss': state_nll_loss.item(),
            'reward_nll_loss': reward_nll_loss.item(),
            'done_loss': done_loss.item(),
            'std_regularization': std_regularization.item(),
            'avg_state_uncertainty': torch.mean(next_state_logstd.exp()).item(),
            'avg_reward_uncertainty': torch.mean(reward_logstd.exp()).item()
        }
    
    def train_from_buffer(self, replay_buffer, batch_size: int = 256, epochs: int = 1) -> Dict[str, float]:
        """
        从经验回放缓冲区训练概率世界模型
        
        Args:
            replay_buffer: 经验回放缓冲区
            batch_size: 批次大小
            epochs: 训练轮数
            
        Returns:
            avg_losses: 平均损失
        """
        if len(replay_buffer) < batch_size:
            return None
        
        total_losses = {
            'total_loss': 0, 'state_nll_loss': 0, 'reward_nll_loss': 0,
            'done_loss': 0, 'std_regularization': 0,
            'avg_state_uncertainty': 0, 'avg_reward_uncertainty': 0
        }
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
    
    def evaluate_uncertainty(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, float]:
        """
        评估模型预测的不确定性
        
        Args:
            states: 状态
            actions: 动作
            
        Returns:
            uncertainty_info: 不确定性信息
        """
        with torch.no_grad():
            next_state_mean, next_state_logstd, reward_mean, reward_logstd, pred_dones = self.world_model(states, actions)
            
            return {
                'state_uncertainty': torch.mean(next_state_logstd.exp()).item(),
                'reward_uncertainty': torch.mean(reward_logstd.exp()).item(),
                'max_state_uncertainty': torch.max(next_state_logstd.exp()).item(),
                'max_reward_uncertainty': torch.max(reward_logstd.exp()).item(),
                'uncertainty_ratio': torch.mean(reward_logstd.exp()).item() / torch.mean(next_state_logstd.exp()).item()
            }
    
    def is_uncertain(self, states: torch.Tensor, actions: torch.Tensor) -> bool:
        """
        判断模型是否对当前输入不确定
        
        Args:
            states: 状态
            actions: 动作
            
        Returns:
            is_uncertain: 是否不确定
        """
        uncertainty_info = self.evaluate_uncertainty(states, actions)
        return uncertainty_info['state_uncertainty'] > self.uncertainty_threshold


class UncertaintyAwareRewardShaping:
    """
    不确定性感知的奖励塑造
    """
    
    def __init__(self, uncertainty_weight: float = 0.1, 
                 exploration_bonus: float = 0.5,
                 safety_penalty: float = 1.0):
        self.uncertainty_weight = uncertainty_weight
        self.exploration_bonus = exploration_bonus
        self.safety_penalty = safety_penalty
    
    def calculate_intrinsic_reward(self, uncertainty_info: Dict[str, float], 
                                 safety_score: float) -> float:
        """
        计算内在奖励
        
        Args:
            uncertainty_info: 不确定性信息
            safety_score: 安全分数
            
        Returns:
            intrinsic_reward: 内在奖励
        """
        # 探索奖励：鼓励探索高不确定性区域
        exploration_reward = self.exploration_bonus * uncertainty_info['state_uncertainty']
        
        # 安全奖励：鼓励安全行为
        safety_reward = safety_score * self.safety_penalty
        
        # 综合内在奖励
        intrinsic_reward = exploration_reward + safety_reward
        
        return intrinsic_reward
    
    def calculate_penalty_for_uncertainty(self, uncertainty_info: Dict[str, float]) -> float:
        """
        计算不确定性惩罚
        
        Args:
            uncertainty_info: 不确定性信息
            
        Returns:
            uncertainty_penalty: 不确定性惩罚
        """
        # 如果不确定性过高，给予惩罚
        if uncertainty_info['state_uncertainty'] > 1.0:  # 高不确定性阈值
            return self.uncertainty_weight * uncertainty_info['state_uncertainty']
        else:
            return 0.0


class AdaptiveTrainingScheduler:
    """
    自适应训练调度器
    """
    
    def __init__(self, initial_lr: float = 1e-3, 
                 uncertainty_threshold: float = 0.5,
                 lr_decay_factor: float = 0.9,
                 min_lr: float = 1e-5):
        self.initial_lr = initial_lr
        self.uncertainty_threshold = uncertainty_threshold
        self.lr_decay_factor = lr_decay_factor
        self.min_lr = min_lr
        self.current_lr = initial_lr
        
    def update_learning_rate(self, optimizer: torch.optim.Optimizer, 
                           uncertainty_info: Dict[str, float]):
        """
        根据不确定性更新学习率
        
        Args:
            optimizer: 优化器
            uncertainty_info: 不确定性信息
        """
        # 如果不确定性过高，降低学习率
        if uncertainty_info['state_uncertainty'] > self.uncertainty_threshold:
            self.current_lr = max(self.min_lr, self.current_lr * self.lr_decay_factor)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_lr
    
    def should_increase_exploration(self, uncertainty_info: Dict[str, float]) -> bool:
        """
        判断是否应该增加探索
        
        Args:
            uncertainty_info: 不确定性信息
            
        Returns:
            should_explore: 是否应该增加探索
        """
        return uncertainty_info['state_uncertainty'] < self.uncertainty_threshold * 0.5
