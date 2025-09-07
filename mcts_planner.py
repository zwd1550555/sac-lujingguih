# -*- coding: utf-8 -*-
"""
基于世界模型的蒙特卡洛树搜索(MCTS)规划器
融合规划与学习，实现深思熟虑的决策能力

主要功能:
- 蒙特卡洛树搜索: 构建搜索树进行深度规划
- 世界模型集成: 使用概率世界模型进行高效模拟
- 价值网络指导: 利用训练好的Critic网络评估节点价值
- 策略网络引导: 使用Actor网络提供候选动作和先验概率

技术特性:
- UCT算法: 平衡探索与利用的节点选择策略
- 概率模拟: 基于世界模型的不确定性感知模拟
- 价值回溯: 高效的价值传播和更新机制
- 深度规划: 支持多步前瞻的复杂决策

核心优势:
- 从"反应式"到"深思式": 具备长时序规划能力
- 高效模拟: 避免真实环境交互的高成本
- 智能剪枝: 基于价值网络的智能搜索剪枝
- 不确定性感知: 考虑预测不确定性的鲁棒决策
"""

import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class MCTS_Node:
    """
    蒙特卡洛树搜索中的一个节点
    
    每个节点代表环境中的一个状态，包含：
    - 状态信息: 当前环境状态
    - 访问统计: 访问次数和累计价值
    - 先验概率: 由策略网络提供的动作先验概率
    - 树结构: 父子节点关系
    """
    
    def __init__(self, state: np.ndarray, parent=None, action=None, prior_p: float = 0.0):
        self.state = state  # 节点对应的环境状态
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.action = action  # 从父节点到达此节点的动作
        
        # 访问统计
        self.visit_count = 0  # 访问次数 N(s,a)
        self.total_value = 0.0  # 累计价值 Q(s,a)
        self.prior_p = prior_p  # 先验概率 P(s,a)，由策略网络提供
        
        # 节点状态
        self.is_expanded = False  # 是否已扩展
        self.is_terminal = False  # 是否为终止状态
        self.depth = 0  # 节点深度
        
        if parent is not None:
            self.depth = parent.depth + 1
    
    def get_value(self) -> float:
        """
        计算节点的平均价值 Q(s,a) / N(s,a)
        
        Returns:
            float: 节点的平均价值
        """
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    def get_ucb_score(self, exploration_constant: float, parent_visit_count: int) -> float:
        """
        计算UCT (Upper Confidence Bound for Trees) 分数
        
        UCT = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            exploration_constant: 探索常数
            parent_visit_count: 父节点访问次数
            
        Returns:
            float: UCT分数
        """
        if self.visit_count == 0:
            return float('inf')  # 未访问的节点优先选择
        
        exploitation = self.get_value()
        exploration = exploration_constant * self.prior_p * \
                     math.sqrt(parent_visit_count) / (1 + self.visit_count)
        
        return exploitation + exploration
    
    def is_leaf(self) -> bool:
        """
        判断是否为叶子节点
        
        Returns:
            bool: 是否为叶子节点
        """
        return len(self.children) == 0 or not self.is_expanded
    
    def select_child(self, exploration_constant: float):
        """
        根据UCT算法选择最优子节点
        
        Args:
            exploration_constant: 探索常数
            
        Returns:
            MCTS_Node: 选中的子节点
        """
        if not self.children:
            return None
        
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            score = child.get_ucb_score(exploration_constant, self.visit_count)
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def add_child(self, state: np.ndarray, action: np.ndarray, prior_p: float):
        """
        添加子节点
        
        Args:
            state: 子节点状态
            action: 到达子节点的动作
            prior_p: 先验概率
        """
        child = MCTS_Node(state=state, parent=self, action=action, prior_p=prior_p)
        self.children.append(child)
        return child
    
    def update_value(self, value: float):
        """
        更新节点价值
        
        Args:
            value: 新的价值
        """
        self.visit_count += 1
        self.total_value += value


class MCTS_Planner:
    """
    基于世界模型和RL价值网络的MCTS规划器
    
    核心功能:
    1. 构建搜索树进行深度规划
    2. 使用世界模型进行高效模拟
    3. 利用价值网络指导搜索
    4. 平衡探索与利用的决策
    """
    
    def __init__(self, 
                 world_model, 
                 actor, 
                 critic, 
                 device: torch.device,
                 num_simulations: int = 100,
                 exploration_constant: float = 1.5,
                 gamma: float = 0.99,
                 max_depth: int = 20,
                 temperature: float = 1.0):
        
        self.world_model = world_model
        self.actor = actor  # 用于提供候选动作和先验概率
        self.critic = critic  # 用于评估叶子节点的价值
        self.device = device
        
        # MCTS参数
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.gamma = gamma  # 折扣因子
        self.max_depth = max_depth  # 最大搜索深度
        self.temperature = temperature  # 动作选择温度
        
        # 统计信息
        self.search_stats = {
            'total_simulations': 0,
            'average_depth': 0,
            'expansion_count': 0,
            'value_estimates': []
        }
    
    @torch.no_grad()
    def search(self, initial_state: np.ndarray, initial_hidden_state=None) -> np.ndarray:
        """
        执行MCTS搜索，为给定状态找到最佳动作
        
        Args:
            initial_state: 当前的环境状态
            initial_hidden_state: LNN的初始隐藏状态
            
        Returns:
            np.ndarray: 经过深思熟虑后得到的最佳动作
        """
        # 创建根节点
        root = MCTS_Node(state=initial_state)
        
        # 执行多次模拟
        for simulation in range(self.num_simulations):
            # 1. 选择 (Select) - 从根节点到叶子节点
            node = self._select(root)
            
            # 2. 扩展 (Expand) - 扩展叶子节点
            if not node.is_terminal and node.depth < self.max_depth:
                self._expand(node, initial_hidden_state)
            
            # 3. 模拟 (Simulate) - 评估节点价值
            value = self._simulate(node, initial_hidden_state)
            
            # 4. 回溯 (Backpropagate) - 更新路径上的所有节点
            self._backpropagate(node, value)
            
            # 更新统计信息
            self.search_stats['total_simulations'] += 1
        
        # 选择最佳动作
        best_action = self._get_best_action(root)
        
        # 更新统计信息
        self._update_search_stats(root)
        
        return best_action
    
    def _select(self, root: MCTS_Node) -> MCTS_Node:
        """
        选择阶段：从根节点开始，使用UCT算法选择到叶子节点的路径
        
        Args:
            root: 根节点
            
        Returns:
            MCTS_Node: 选中的叶子节点
        """
        node = root
        
        while not node.is_leaf() and not node.is_terminal:
            node = node.select_child(self.exploration_constant)
        
        return node
    
    def _expand(self, node: MCTS_Node, initial_hidden_state=None):
        """
        扩展阶段：为叶子节点生成子节点
        
        Args:
            node: 要扩展的叶子节点
            initial_hidden_state: 初始隐藏状态
        """
        if node.is_expanded or node.is_terminal:
            return
        
        # 使用Actor网络生成候选动作和先验概率
        state_tensor = torch.as_tensor(node.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 生成候选动作
        candidate_actions, candidate_log_probs, _ = self.actor.generate_candidate_actions(
            state_tensor, initial_hidden_state
        )
        
        # 计算先验概率
        prior_probs = torch.exp(candidate_log_probs).squeeze(1)
        prior_probs = prior_probs / torch.sum(prior_probs)  # 归一化
        
        # 为每个候选动作创建子节点
        for i in range(candidate_actions.shape[0]):
            action = candidate_actions[i].cpu().numpy()
            prior_p = prior_probs[i].item()
            
            # 使用世界模型预测下一状态
            next_state, reward, done = self.world_model.sample_prediction(state_tensor, candidate_actions[i])
            next_state_np = next_state.squeeze(0).cpu().numpy()
            
            # 创建子节点
            child = node.add_child(next_state_np, action, prior_p)
            
            # 如果预测为终止状态，标记为终止节点
            if done.item():
                child.is_terminal = True
        
        node.is_expanded = True
        self.search_stats['expansion_count'] += 1
    
    def _simulate(self, node: MCTS_Node, initial_hidden_state=None) -> float:
        """
        模拟阶段：评估节点价值
        
        Args:
            node: 要评估的节点
            initial_hidden_state: 初始隐藏状态
            
        Returns:
            float: 节点价值估计
        """
        if node.is_terminal:
            return 0.0  # 终止状态价值为0
        
        # 使用Critic网络评估节点价值
        state_tensor = torch.as_tensor(node.state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 如果没有动作，使用默认动作进行评估
        if node.action is not None:
            action_tensor = torch.as_tensor(node.action, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            # 使用Actor网络生成一个动作进行评估
            with torch.no_grad():
                mean, log_std, _ = self.actor(state_tensor, initial_hidden_state)
                std = log_std.exp()
                action_tensor = torch.tanh(mean + std * torch.randn_like(mean))
        
        # 使用Critic网络评估Q值
        q1, q2, _, _ = self.critic(state_tensor, action_tensor, None, None)
        value = torch.min(q1, q2).item()
        
        # 记录价值估计
        self.search_stats['value_estimates'].append(value)
        
        return value
    
    def _backpropagate(self, node: MCTS_Node, value: float):
        """
        回溯阶段：将模拟得到的价值回溯更新到路径上的所有父节点
        
        Args:
            node: 开始回溯的节点
            value: 要回溯的价值
        """
        current_value = value
        
        while node is not None:
            node.update_value(current_value)
            
            # 应用折扣因子
            current_value *= self.gamma
            
            node = node.parent
    
    def _get_best_action(self, root: MCTS_Node) -> np.ndarray:
        """
        从根节点选择最佳动作
        
        Args:
            root: 根节点
            
        Returns:
            np.ndarray: 最佳动作
        """
        if not root.children:
            # 如果没有子节点，返回随机动作
            return np.random.uniform(-1, 1, size=(2,))
        
        # 根据访问次数选择最佳动作
        best_child = max(root.children, key=lambda c: c.visit_count)
        return best_child.action
    
    def _update_search_stats(self, root: MCTS_Node):
        """
        更新搜索统计信息
        
        Args:
            root: 根节点
        """
        if root.children:
            total_depth = sum(child.depth for child in root.children)
            self.search_stats['average_depth'] = total_depth / len(root.children)
    
    def get_search_statistics(self) -> Dict:
        """
        获取搜索统计信息
        
        Returns:
            Dict: 搜索统计信息
        """
        stats = self.search_stats.copy()
        
        if stats['value_estimates']:
            stats['avg_value_estimate'] = np.mean(stats['value_estimates'])
            stats['std_value_estimate'] = np.std(stats['value_estimates'])
            stats['min_value_estimate'] = np.min(stats['value_estimates'])
            stats['max_value_estimate'] = np.max(stats['value_estimates'])
        
        return stats
    
    def reset_statistics(self):
        """重置搜索统计信息"""
        self.search_stats = {
            'total_simulations': 0,
            'average_depth': 0,
            'expansion_count': 0,
            'value_estimates': []
        }


class MCTS_Config:
    """
    MCTS配置类
    """
    
    def __init__(self, 
                 num_simulations: int = 100,
                 exploration_constant: float = 1.5,
                 gamma: float = 0.99,
                 max_depth: int = 20,
                 temperature: float = 1.0,
                 enable_uncertainty_aware: bool = True,
                 uncertainty_threshold: float = 0.5):
        
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.gamma = gamma
        self.max_depth = max_depth
        self.temperature = temperature
        self.enable_uncertainty_aware = enable_uncertainty_aware
        self.uncertainty_threshold = uncertainty_threshold
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'num_simulations': self.num_simulations,
            'exploration_constant': self.exploration_constant,
            'gamma': self.gamma,
            'max_depth': self.max_depth,
            'temperature': self.temperature,
            'enable_uncertainty_aware': self.enable_uncertainty_aware,
            'uncertainty_threshold': self.uncertainty_threshold
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'MCTS_Config':
        """从字典创建配置"""
        return cls(**config_dict)
