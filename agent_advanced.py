# -*- coding: utf-8 -*-
"""
高级智能体实现 - 集成因果推理、主动决策和MCTS规划
实现基于干预预测的主动决策系统和深思熟虑的规划能力

主要特性:
- 基于干预预测的主动决策: 在决策前利用因果模型进行长期规划
- 候选动作评估: 生成多个候选动作并评估其长期效果
- 综合评分系统: 结合奖励、安全性和不确定性的综合评分
- 长期规划能力: 具备"牺牲眼前利益换取长远优势"的决策能力
- MCTS规划: 基于世界模型的蒙特卡洛树搜索，实现深思熟虑的决策

技术优势:
- 从短视决策到长期规划
- 从被动反应到主动预测
- 从单一评估到多维综合评估
- 从确定性决策到考虑不确定性的鲁棒决策
- 从"反应式"到"深思式"的决策能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from agent_lnn import LiquidSACAgent, LiquidActor, LiquidCritic, SequenceReplayBuffer
from world_model import WorldModel, CausalReasoner
from mcts_planner import MCTS_Planner, MCTS_Config


class AdvancedLiquidActor(LiquidActor):
    """
    高级液态神经网络Actor - 支持候选动作生成
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, 
                 log_std_min: float = -20, log_std_max: float = 2,
                 num_candidates: int = 5):
        super().__init__(state_dim, action_dim, hidden_dim, log_std_min, log_std_max)
        self.num_candidates = num_candidates
        
    def generate_candidate_actions(self, state: torch.Tensor, hidden_state=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成多个候选动作
        
        Args:
            state: 当前状态
            hidden_state: 隐藏状态
            
        Returns:
            candidate_actions: 候选动作列表 (num_candidates, action_dim)
            candidate_log_probs: 候选动作的对数概率
        """
        if state.ndim == 2:
            state = state.unsqueeze(0)
        
        # 生成基础动作分布
        mean, log_std, new_hidden_state = super().forward(state, hidden_state)
        std = log_std.exp()
        
        # 生成多个候选动作
        candidate_actions = []
        candidate_log_probs = []
        
        for _ in range(self.num_candidates):
            # 从分布中采样
            z = torch.randn_like(mean)
            action = torch.tanh(mean + std * z)
            
            # 计算对数概率
            log_prob = -0.5 * (z ** 2 + 2 * log_std + np.log(2 * np.pi))
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
            candidate_actions.append(action)
            candidate_log_probs.append(log_prob)
        
        candidate_actions = torch.cat(candidate_actions, dim=0)  # (num_candidates, action_dim)
        candidate_log_probs = torch.cat(candidate_log_probs, dim=0)  # (num_candidates, 1)
        
        return candidate_actions, candidate_log_probs, new_hidden_state


class InterventionBasedDecisionMaker:
    """
    基于干预预测的决策制定器
    """
    
    def __init__(self, causal_reasoner: CausalReasoner, 
                 reward_weight: float = 0.7, 
                 safety_weight: float = 0.3,
                 uncertainty_weight: float = 0.1,
                 prediction_horizon: int = 10):
        self.causal_reasoner = causal_reasoner
        self.reward_weight = reward_weight
        self.safety_weight = safety_weight
        self.uncertainty_weight = uncertainty_weight
        self.prediction_horizon = prediction_horizon
        
    def evaluate_candidate_actions(self, state: torch.Tensor, 
                                 candidate_actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        评估候选动作的长期效果
        
        Args:
            state: 当前状态
            candidate_actions: 候选动作 (num_candidates, action_dim)
            
        Returns:
            evaluation_results: 包含各种评估指标的字典
        """
        num_candidates = candidate_actions.shape[0]
        total_rewards = []
        safety_scores = []
        uncertainty_scores = []
        
        for i in range(num_candidates):
            action = candidate_actions[i]
            
            # 进行干预预测
            prediction = self.causal_reasoner.intervention_prediction(
                state, action, horizon=self.prediction_horizon
            )
            
            # 提取评估指标
            total_reward = prediction['total_reward']
            safety_score = prediction['safety_score']
            
            # 计算不确定性（基于预测的方差）
            uncertainty = self._calculate_uncertainty(prediction)
            
            total_rewards.append(total_reward)
            safety_scores.append(safety_score)
            uncertainty_scores.append(uncertainty)
        
        return {
            'total_rewards': torch.tensor(total_rewards),
            'safety_scores': torch.tensor(safety_scores),
            'uncertainty_scores': torch.tensor(uncertainty_scores)
        }
    
    def _calculate_uncertainty(self, prediction: Dict) -> float:
        """
        计算预测的不确定性
        
        Args:
            prediction: 预测结果字典
            
        Returns:
            uncertainty: 不确定性分数
        """
        # 基于预测状态的方差计算不确定性
        states = prediction.get('states', [])
        if len(states) > 1:
            # 计算状态变化的方差
            state_changes = []
            for i in range(1, len(states)):
                change = torch.norm(states[i] - states[i-1])
                state_changes.append(change.item())
            
            if state_changes:
                uncertainty = np.var(state_changes)
            else:
                uncertainty = 0.0
        else:
            uncertainty = 0.0
            
        return uncertainty
    
    def select_best_action(self, candidate_actions: torch.Tensor, 
                          evaluation_results: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """
        基于综合评分选择最佳动作
        
        Args:
            candidate_actions: 候选动作
            evaluation_results: 评估结果
            
        Returns:
            best_action: 最佳动作
            best_index: 最佳动作的索引
        """
        total_rewards = evaluation_results['total_rewards']
        safety_scores = evaluation_results['safety_scores']
        uncertainty_scores = evaluation_results['uncertainty_scores']
        
        # 归一化评分
        total_rewards_norm = F.softmax(total_rewards, dim=0)
        safety_scores_norm = F.softmax(safety_scores, dim=0)
        uncertainty_scores_norm = F.softmax(-uncertainty_scores, dim=0)  # 负号表示不确定性越小越好
        
        # 综合评分
        composite_scores = (self.reward_weight * total_rewards_norm + 
                          self.safety_weight * safety_scores_norm + 
                          self.uncertainty_weight * uncertainty_scores_norm)
        
        # 选择最佳动作
        best_index = torch.argmax(composite_scores).item()
        best_action = candidate_actions[best_index]
        
        return best_action, best_index


class AdvancedLiquidSACAgent(LiquidSACAgent):
    """
    高级液态神经网络SAC智能体 - 集成因果推理、主动决策和MCTS规划
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 replay_buffer_capacity: int = 1_000_000, actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.005,
                 alpha: float = 0.2, target_update_interval: int = 1,
                 sequence_length: int = 10, num_candidates: int = 5,
                 use_intervention_decision: bool = True,
                 use_mcts_planning: bool = False,
                 mcts_config: Optional[MCTS_Config] = None):
        
        # 初始化父类
        super().__init__(state_dim, action_dim, hidden_dim, replay_buffer_capacity,
                        actor_lr, critic_lr, gamma, tau, alpha, target_update_interval,
                        sequence_length)
        
        # 替换为高级Actor
        self.actor = AdvancedLiquidActor(state_dim, action_dim, hidden_dim, 
                                       num_candidates=num_candidates).to(self.device)
        
        # 重新初始化优化器
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 决策相关参数
        self.use_intervention_decision = use_intervention_decision
        self.use_mcts_planning = use_mcts_planning
        self.num_candidates = num_candidates
        self.decision_maker = None  # 将在外部设置
        self.mcts_planner = None  # MCTS规划器
        
        # MCTS配置
        self.mcts_config = mcts_config if mcts_config is not None else MCTS_Config()
        
    def set_decision_maker(self, decision_maker: InterventionBasedDecisionMaker):
        """
        设置决策制定器
        
        Args:
            decision_maker: 基于干预预测的决策制定器
        """
        self.decision_maker = decision_maker
    
    def set_mcts_planner(self, world_model, critic):
        """
        设置MCTS规划器
        
        Args:
            world_model: 世界模型
            critic: Critic网络
        """
        if self.use_mcts_planning:
            self.mcts_planner = MCTS_Planner(
                world_model=world_model,
                actor=self.actor,
                critic=critic,
                device=self.device,
                num_simulations=self.mcts_config.num_simulations,
                exploration_constant=self.mcts_config.exploration_constant,
                gamma=self.mcts_config.gamma,
                max_depth=self.mcts_config.max_depth,
                temperature=self.mcts_config.temperature
            )
            print("MCTS Planner has been initialized.")
    
    def set_mcts_config(self, mcts_config: MCTS_Config):
        """
        设置MCTS配置
        
        Args:
            mcts_config: MCTS配置
        """
        self.mcts_config = mcts_config
        if self.mcts_planner is not None:
            # 更新现有规划器的配置
            self.mcts_planner.num_simulations = mcts_config.num_simulations
            self.mcts_planner.exploration_constant = mcts_config.exploration_constant
            self.mcts_planner.gamma = mcts_config.gamma
            self.mcts_planner.max_depth = mcts_config.max_depth
            self.mcts_planner.temperature = mcts_config.temperature
    
    @torch.no_grad()
    def select_action(self, state: np.ndarray, hidden_state=None, evaluate: bool = False) -> Tuple[np.ndarray, any]:
        """
        选择动作 - 支持MCTS规划、基于干预预测的主动决策和标准决策
        
        Args:
            state: 当前状态
            hidden_state: 隐藏状态
            evaluate: 是否为评估模式
            
        Returns:
            action: 选择的动作
            new_hidden_state: 新的隐藏状态
        """
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # --- MCTS规划分支 ---
        if self.use_mcts_planning and not evaluate and self.mcts_planner is not None:
            # 1. 运行MCTS搜索来获得深思熟虑后的最佳动作
            best_action = self.mcts_planner.search(initial_state=state, initial_hidden_state=hidden_state)
            
            # 2. MCTS规划本身不更新LNN的隐藏状态。
            # 为了保持隐藏状态的连续性，我们仍然需要通过LNN正向传播一次来获得新的隐藏状态。
            # 这是一个工程上的权衡，确保了时序记忆的连贯性。
            _, _, new_hidden_state = self.actor(state_t, hidden_state)
            
            return best_action, new_hidden_state
        
        # --- 基于干预预测的主动决策分支 ---
        if self.use_intervention_decision and not evaluate and self.decision_maker is not None:
            # 使用基于干预预测的主动决策
            candidate_actions, candidate_log_probs, new_hidden_state = self.actor.generate_candidate_actions(state_t, hidden_state)
            
            # 评估候选动作
            evaluation_results = self.decision_maker.evaluate_candidate_actions(state_t, candidate_actions)
            
            # 选择最佳动作
            best_action, best_index = self.decision_maker.select_best_action(candidate_actions, evaluation_results)
            
            return best_action.squeeze(0).cpu().numpy(), new_hidden_state
        
        # --- 默认的标准SAC决策 ---
        # 调用父类的标准select_action
        action_np, new_hidden_state = super().select_action(state, hidden_state, evaluate)
        return action_np, new_hidden_state
    
    def get_decision_info(self, state: np.ndarray, hidden_state=None) -> Dict:
        """
        获取决策信息 - 用于调试和分析
        
        Args:
            state: 当前状态
            hidden_state: 隐藏状态
            
        Returns:
            decision_info: 决策信息字典
        """
        decision_info = {
            'decision_mode': 'standard',
            'use_mcts_planning': self.use_mcts_planning,
            'use_intervention_decision': self.use_intervention_decision
        }
        
        # MCTS规划信息
        if self.use_mcts_planning and self.mcts_planner is not None:
            decision_info['decision_mode'] = 'mcts_planning'
            decision_info['mcts_stats'] = self.mcts_planner.get_search_statistics()
            decision_info['mcts_config'] = self.mcts_config.to_dict()
        
        # 干预预测决策信息
        elif self.use_intervention_decision and self.decision_maker is not None:
            decision_info['decision_mode'] = 'intervention_decision'
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            candidate_actions, _, _ = self.actor.generate_candidate_actions(state_t, hidden_state)
            evaluation_results = self.decision_maker.evaluate_candidate_actions(state_t, candidate_actions)
            
            decision_info.update({
                'candidate_actions': candidate_actions.cpu().numpy(),
                'evaluation_results': {k: v.cpu().numpy() for k, v in evaluation_results.items()},
                'num_candidates': self.num_candidates
            })
        
        return decision_info
    
    def get_mcts_statistics(self) -> Dict:
        """
        获取MCTS统计信息
        
        Returns:
            Dict: MCTS统计信息
        """
        if self.mcts_planner is not None:
            return self.mcts_planner.get_search_statistics()
        return {}
    
    def reset_mcts_statistics(self):
        """重置MCTS统计信息"""
        if self.mcts_planner is not None:
            self.mcts_planner.reset_statistics()


class ProbabilisticWorldModel(WorldModel):
    """
    概率世界模型 - 输出预测的不确定性
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(state_dim, action_dim, hidden_dim)
        
        # 修改预测器以输出概率分布
        self.transition_predictor_mean = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.transition_predictor_logstd = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.reward_predictor_mean = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.reward_predictor_logstd = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        前向传播 - 输出概率分布
        
        Args:
            state: 当前状态
            action: 当前动作
            
        Returns:
            next_state_mean: 下一状态的均值
            next_state_logstd: 下一状态的对数标准差
            reward_mean: 奖励的均值
            reward_logstd: 奖励的对数标准差
            done: 终止状态预测
        """
        # 编码状态和动作
        state_encoded = self.state_encoder(state)
        action_encoded = self.action_encoder(action)
        
        # 拼接状态和动作编码
        combined = torch.cat([state_encoded, action_encoded], dim=1)
        
        # 预测下一状态的概率分布
        next_state_mean = self.transition_predictor_mean(combined)
        next_state_logstd = self.transition_predictor_logstd(combined)
        next_state_logstd = torch.clamp(next_state_logstd, -10, 2)  # 限制标准差范围
        
        # 预测奖励的概率分布
        reward_mean = self.reward_predictor_mean(combined)
        reward_logstd = self.reward_predictor_logstd(combined)
        reward_logstd = torch.clamp(reward_logstd, -10, 2)
        
        # 预测终止状态
        done = self.done_predictor(combined).squeeze()
        
        return next_state_mean, next_state_logstd, reward_mean, reward_logstd, done
    
    def sample_prediction(self, state: torch.Tensor, action: torch.Tensor):
        """
        从概率分布中采样预测结果
        
        Args:
            state: 当前状态
            action: 当前动作
            
        Returns:
            next_state: 采样的下一状态
            reward: 采样的奖励
            done: 终止状态预测
        """
        next_state_mean, next_state_logstd, reward_mean, reward_logstd, done = self.forward(state, action)
        
        # 从分布中采样
        next_state_std = next_state_logstd.exp()
        reward_std = reward_logstd.exp()
        
        next_state = next_state_mean + next_state_std * torch.randn_like(next_state_mean)
        reward = reward_mean + reward_std * torch.randn_like(reward_mean)
        
        return next_state, reward, done


class ProbabilisticCausalReasoner(CausalReasoner):
    """
    概率因果推理器 - 处理预测不确定性
    """
    
    def __init__(self, world_model: ProbabilisticWorldModel, device: torch.device):
        super().__init__(world_model, device)
        self.world_model = world_model
    
    def intervention_prediction_with_uncertainty(self, state: torch.Tensor, 
                                               intervention_action: torch.Tensor, 
                                               horizon: int = 5,
                                               num_samples: int = 10) -> Dict:
        """
        带不确定性的干预预测
        
        Args:
            state: 当前状态
            intervention_action: 干预动作
            horizon: 预测时间范围
            num_samples: 采样次数
            
        Returns:
            prediction: 包含不确定性的预测结果
        """
        predictions = []
        
        for _ in range(num_samples):
            current_state = state
            states = [current_state]
            rewards = []
            dones = []
            
            for t in range(horizon):
                # 从概率分布中采样预测
                next_state, reward, done = self.world_model.sample_prediction(current_state, intervention_action)
                
                states.append(next_state)
                rewards.append(reward.item())
                dones.append(done.item())
                
                if done.item():
                    break
                    
                current_state = next_state
            
            predictions.append({
                'states': states[1:],
                'rewards': rewards,
                'dones': dones,
                'total_reward': sum(rewards)
            })
        
        # 计算统计信息
        total_rewards = [p['total_reward'] for p in predictions]
        safety_scores = [self._calculate_safety_score(p['states']) for p in predictions]
        
        return {
            'mean_total_reward': np.mean(total_rewards),
            'std_total_reward': np.std(total_rewards),
            'mean_safety_score': np.mean(safety_scores),
            'std_safety_score': np.std(safety_scores),
            'uncertainty': np.std(total_rewards) + np.std(safety_scores),  # 综合不确定性
            'predictions': predictions
        }
    
    def _calculate_safety_score(self, states: List[torch.Tensor]) -> float:
        """
        计算安全分数
        
        Args:
            states: 预测的状态序列
            
        Returns:
            safety_score: 安全分数
        """
        if not states:
            return 0.0
        
        safety_scores = []
        for state in states:
            # 简化的安全评估：检查速度是否在安全范围内
            speed = abs(state[0].item())  # 归一化速度
            if speed < 0.8:  # 安全速度范围
                safety_scores.append(1.0)
            else:
                safety_scores.append(max(0.0, 1.0 - (speed - 0.8) * 2))
        
        return np.mean(safety_scores) if safety_scores else 0.0
