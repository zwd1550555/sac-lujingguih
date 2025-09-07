# -*- coding: utf-8 -*-
"""
高级奖励设计系统
实现舒适性和能效奖励，引导更精细的驾驶行为

主要功能:
- 舒适性奖励: 基于加速度变化率(Jerk)的平滑驾驶奖励
- 能效奖励: 基于履带速度差的能耗优化奖励
- 多维度评估: 结合安全性、效率、舒适性和能效的综合评估
- 自适应权重: 根据训练阶段动态调整奖励权重

技术特性:
- 物理约束: 基于真实车辆动力学的奖励设计
- 平滑性评估: 评估驾驶行为的平滑程度
- 能耗优化: 优化履带车的能耗效率
- 行为引导: 引导智能体学习专业驾驶员的行为模式
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import deque


class ComfortabilityAnalyzer:
    """
    舒适性分析器
    """
    
    def __init__(self, jerk_window_size: int = 10, jerk_threshold: float = 2.0):
        self.jerk_window_size = jerk_window_size
        self.jerk_threshold = jerk_threshold
        self.acceleration_history = deque(maxlen=jerk_window_size)
        self.jerk_history = deque(maxlen=jerk_window_size)
        
    def calculate_jerk(self, current_acceleration: float, dt: float = 0.1) -> float:
        """
        计算加速度变化率(Jerk)
        
        Args:
            current_acceleration: 当前加速度
            dt: 时间步长
            
        Returns:
            jerk: 加速度变化率
        """
        self.acceleration_history.append(current_acceleration)
        
        if len(self.acceleration_history) < 2:
            return 0.0
        
        # 计算加速度变化率
        jerk = (current_acceleration - self.acceleration_history[-2]) / dt
        self.jerk_history.append(abs(jerk))
        
        return jerk
    
    def get_comfortability_score(self) -> float:
        """
        获取舒适性分数
        
        Returns:
            comfort_score: 舒适性分数 (0-1)
        """
        if not self.jerk_history:
            return 1.0
        
        # 基于Jerk的舒适性评估
        avg_jerk = np.mean(self.jerk_history)
        max_jerk = np.max(self.jerk_history)
        
        # 舒适性分数：Jerk越小，分数越高
        comfort_score = max(0.0, 1.0 - (avg_jerk / self.jerk_threshold))
        
        # 惩罚极端Jerk值
        if max_jerk > self.jerk_threshold * 2:
            comfort_score *= 0.5
        
        return comfort_score
    
    def get_jerk_penalty(self) -> float:
        """
        获取Jerk惩罚
        
        Returns:
            jerk_penalty: Jerk惩罚值
        """
        if not self.jerk_history:
            return 0.0
        
        avg_jerk = np.mean(self.jerk_history)
        return max(0.0, avg_jerk - self.jerk_threshold * 0.5)


class EnergyEfficiencyAnalyzer:
    """
    【优化后】能效分析器：利用动力学模型参数计算更真实的能耗
    """
    def __init__(self, efficiency_window_size: int = 20):
        self.efficiency_window_size = efficiency_window_size
        self.energy_consumption_history = deque(maxlen=efficiency_window_size)
        self.last_power_W = 0.0

    def calculate_energy_consumption(self, ego_state: Dict, dynamics, dt: float) -> float:
        """
        根据物理模型计算瞬时功耗并估算能耗。

        Args:
            ego_state (Dict): 主车当前状态 (包含 v_mps, acc_mpss)
            dynamics: 车辆动力学参数对象（VehicleDynamics）
            dt (float): 时间步长

        Returns:
            float: 当前时间步的估算能耗 (焦耳)
        """
        v = float(ego_state.get('v_mps', 0.0))
        acc = float(ego_state.get('acc_mpss', 0.0))

        # 驱动功率：P_drive = (m*a) * v
        F_net = dynamics.mass * acc
        P_drive = F_net * v

        # 阻力功率：P_resist = (F_rolling + F_drag) * v
        F_rolling = dynamics.coeff_rolling * dynamics.total_weight
        F_drag = dynamics.coeff_drag * v * v
        P_resist = (F_rolling + F_drag) * v

        # 仅计正功率消耗
        P_total = max(0.0, P_resist) + max(0.0, P_drive)
        energy_consumption_J = ((self.last_power_W + P_total) / 2.0) * dt
        self.last_power_W = P_total
        self.energy_consumption_history.append(energy_consumption_J)
        return energy_consumption_J

    def get_efficiency_score(self) -> float:
        if not self.energy_consumption_history:
            return 1.0
        avg_energy_kJ = float(np.mean(self.energy_consumption_history)) / 1000.0
        return max(0.0, 1.0 - (avg_energy_kJ / 50.0))

    def get_energy_penalty(self) -> float:
        if not self.energy_consumption_history:
            return 0.0
        avg_energy_kJ = float(np.mean(self.energy_consumption_history)) / 1000.0
        return max(0.0, avg_energy_kJ - 30.0)


class AdvancedRewardCalculator:
    """
    高级奖励计算器
    """
    
    def __init__(self, 
                 comfort_weight: float = 0.2,
                 efficiency_weight: float = 0.15,
                 safety_weight: float = 0.4,
                 goal_weight: float = 0.25,
                 jerk_penalty_weight: float = 0.1,
                 energy_penalty_weight: float = 0.05):
        
        self.comfort_weight = comfort_weight
        self.efficiency_weight = efficiency_weight
        self.safety_weight = safety_weight
        self.goal_weight = goal_weight
        self.jerk_penalty_weight = jerk_penalty_weight
        self.energy_penalty_weight = energy_penalty_weight
        
        # 初始化分析器
        self.comfort_analyzer = ComfortabilityAnalyzer()
        self.energy_analyzer = EnergyEfficiencyAnalyzer()
        
        # 历史记录
        self.reward_history = deque(maxlen=100)
        self.component_rewards = {
            'comfort': deque(maxlen=100),
            'efficiency': deque(maxlen=100),
            'safety': deque(maxlen=100),
            'goal': deque(maxlen=100)
        }
        
    def calculate_advanced_reward(self, 
                                curr_obs: Dict, 
                                prev_obs: Dict, 
                                action: np.ndarray,
                                reward_cfg: Dict) -> Tuple[float, Dict]:
        """
        计算高级奖励
        
        Args:
            curr_obs: 当前观察
            prev_obs: 前一步观察
            action: 当前动作
            reward_cfg: 奖励配置
            
        Returns:
            total_reward: 总奖励
            reward_components: 奖励组件详情
        """
        ego_curr = curr_obs['vehicle_info']['ego']
        ego_prev = prev_obs['vehicle_info']['ego']
        
        # 提取动作信息
        v_left, v_right = action[0], action[1]
        
        # 计算加速度
        prev_speed = float(ego_prev['v_mps'])
        curr_speed = float(ego_curr['v_mps'])
        dt = curr_obs['test_setting']['dt']
        acceleration = (curr_speed - prev_speed) / dt if dt > 0 else 0.0
        
        # 1. 舒适性奖励
        jerk = self.comfort_analyzer.calculate_jerk(acceleration, dt)
        comfort_score = self.comfort_analyzer.get_comfortability_score()
        jerk_penalty = self.comfort_analyzer.get_jerk_penalty()
        comfort_reward = comfort_score * 10.0 - jerk_penalty * self.jerk_penalty_weight
        
        # 2. 能效奖励（使用物理动力学模型估算能耗）
        try:
            dynamics = curr_obs.get('env_dynamics', None)
        except Exception:
            dynamics = None
        if dynamics is not None:
            ego_curr_full = {'v_mps': float(ego_curr['v_mps']), 'acc_mpss': acceleration}
            energy_consumption = self.energy_analyzer.calculate_energy_consumption(ego_curr_full, dynamics, dt)
        else:
            energy_consumption = 0.0
        efficiency_score = self.energy_analyzer.get_efficiency_score()
        energy_penalty = self.energy_analyzer.get_energy_penalty()
        efficiency_reward = efficiency_score * 5.0 - energy_penalty * self.energy_penalty_weight
        
        # 3. 安全性奖励
        safety_reward = self._calculate_safety_reward(curr_obs, prev_obs, reward_cfg)
        
        # 4. 目标奖励
        goal_reward = self._calculate_goal_reward(curr_obs, prev_obs, reward_cfg)
        
        # 计算总奖励
        total_reward = (self.comfort_weight * comfort_reward +
                       self.efficiency_weight * efficiency_reward +
                       self.safety_weight * safety_reward +
                       self.goal_weight * goal_reward)
        
        # 记录奖励组件
        reward_components = {
            'comfort_reward': comfort_reward,
            'efficiency_reward': efficiency_reward,
            'safety_reward': safety_reward,
            'goal_reward': goal_reward,
            'total_reward': total_reward,
            'comfort_score': comfort_score,
            'efficiency_score': efficiency_score,
            'jerk': jerk,
            'energy_consumption': energy_consumption
        }
        
        # 更新历史记录
        self.reward_history.append(total_reward)
        for component, value in reward_components.items():
            if component in self.component_rewards:
                self.component_rewards[component].append(value)
        
        return total_reward, reward_components
    
    def _calculate_safety_reward(self, curr_obs: Dict, prev_obs: Dict, reward_cfg: Dict) -> float:
        """
        计算安全性奖励
        
        Args:
            curr_obs: 当前观察
            prev_obs: 前一步观察
            reward_cfg: 奖励配置
            
        Returns:
            safety_reward: 安全性奖励
        """
        # 碰撞惩罚
        if curr_obs['test_setting']['end'] == 1:  # 碰撞
            return float(reward_cfg.get('collision_penalty', -800.0))
        
        # 安全距离奖励
        ego = curr_obs['vehicle_info']['ego']
        min_dist_to_obs = float('inf')
        
        for vid, vinfo in curr_obs['vehicle_info'].items():
            if vid == 'ego':
                continue
            other_x = float(vinfo['x'])
            other_y = float(vinfo['y'])
            ego_x = float(ego['x'])
            ego_y = float(ego['y'])
            dist = np.hypot(ego_x - other_x, ego_y - other_y)
            min_dist_to_obs = min(min_dist_to_obs, dist)
        
        safety_distance = float(reward_cfg.get('safety_distance', 15.0))
        safety_penalty_weight = float(reward_cfg.get('safety_penalty_weight', 1.0))
        
        if min_dist_to_obs < safety_distance:
            safety_penalty = (1.0 - min_dist_to_obs / safety_distance) * safety_penalty_weight
            return -safety_penalty
        else:
            return 0.0
    
    def _calculate_goal_reward(self, curr_obs: Dict, prev_obs: Dict, reward_cfg: Dict) -> float:
        """
        计算目标奖励
        
        Args:
            curr_obs: 当前观察
            prev_obs: 前一步观察
            reward_cfg: 奖励配置
            
        Returns:
            goal_reward: 目标奖励
        """
        # 到达目标奖励
        if curr_obs['test_setting']['end'] == 0:  # 到达目标
            return float(reward_cfg.get('reach_goal_bonus', 500.0))
        
        # 距离目标奖励
        ego = curr_obs['vehicle_info']['ego']
        goal = curr_obs['test_setting']['goal']
        
        ego_x = float(ego['x'])
        ego_y = float(ego['y'])
        goal_x = float(goal['x'])
        goal_y = float(goal['y'])
        
        dist_to_goal = np.hypot(ego_x - goal_x, ego_y - goal_y)
        
        # 距离奖励：越接近目标奖励越高
        max_distance = 100.0  # 最大距离
        distance_reward = max(0.0, (max_distance - dist_to_goal) / max_distance * 10.0)
        
        return distance_reward
    
    def get_reward_statistics(self) -> Dict:
        """
        获取奖励统计信息
        
        Returns:
            statistics: 奖励统计信息
        """
        if not self.reward_history:
            return {}
        
        statistics = {
            'avg_total_reward': np.mean(self.reward_history),
            'std_total_reward': np.std(self.reward_history),
            'min_total_reward': np.min(self.reward_history),
            'max_total_reward': np.max(self.reward_history)
        }
        
        for component, history in self.component_rewards.items():
            if history:
                statistics[f'avg_{component}'] = np.mean(history)
                statistics[f'std_{component}'] = np.std(history)
        
        return statistics
    
    def update_weights(self, training_phase: str):
        """
        根据训练阶段更新权重
        
        Args:
            training_phase: 训练阶段
        """
        if training_phase == 'basic':
            # 基础阶段：重点关注安全性和目标达成
            self.safety_weight = 0.5
            self.goal_weight = 0.3
            self.comfort_weight = 0.1
            self.efficiency_weight = 0.1
        elif training_phase == 'intermediate':
            # 中级阶段：平衡各项指标
            self.safety_weight = 0.4
            self.goal_weight = 0.25
            self.comfort_weight = 0.2
            self.efficiency_weight = 0.15
        else:  # advanced
            # 高级阶段：重点关注舒适性和能效
            self.safety_weight = 0.3
            self.goal_weight = 0.2
            self.comfort_weight = 0.3
            self.efficiency_weight = 0.2


class RewardVisualizer:
    """
    奖励可视化器
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.component_history = {
            'comfort': deque(maxlen=window_size),
            'efficiency': deque(maxlen=window_size),
            'safety': deque(maxlen=window_size),
            'goal': deque(maxlen=window_size)
        }
    
    def update(self, reward_components: Dict):
        """
        更新奖励历史
        
        Args:
            reward_components: 奖励组件
        """
        self.reward_history.append(reward_components['total_reward'])
        
        for component in self.component_history:
            if component in reward_components:
                self.component_history[component].append(reward_components[component])
    
    def get_visualization_data(self) -> Dict:
        """
        获取可视化数据
        
        Returns:
            visualization_data: 可视化数据
        """
        if not self.reward_history:
            return {}
        
        return {
            'total_rewards': list(self.reward_history),
            'component_rewards': {k: list(v) for k, v in self.component_history.items()},
            'window_size': self.window_size
        }
