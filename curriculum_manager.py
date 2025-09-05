# -*- coding: utf-8 -*-
"""
自动化课程学习管理器
实现智能训练体系，从"手动挡"到"自动挡"

主要功能:
- 自动化课程学习: 根据智能体表现自动调整训练难度
- 动态场景采样: 智能调整不同场景的采样权重
- 性能监控: 持续监控智能体在不同场景下的表现
- 自适应难度: 根据性能自动提升或降低场景难度

技术特性:
- 智能场景选择: 基于性能指标动态调整场景分布
- 难度自适应: 自动调整场景复杂度
- 性能分析: 多维度性能评估
- 训练优化: 自动优化训练效率
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
import json
import os


class ScenarioPerformanceTracker:
    """
    场景性能跟踪器
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.performance_history = defaultdict(lambda: deque(maxlen=window_size))
        self.scenario_stats = defaultdict(lambda: {
            'success_rate': 0.0,
            'collision_rate': 0.0,
            'avg_reward': 0.0,
            'avg_steps': 0.0,
            'episode_count': 0
        })
        
    def update_performance(self, scenario_type: str, episode_result: Dict):
        """
        更新场景性能
        
        Args:
            scenario_type: 场景类型
            episode_result: 回合结果
        """
        # 记录性能历史
        self.performance_history[scenario_type].append(episode_result)
        
        # 更新统计信息
        stats = self.scenario_stats[scenario_type]
        stats['episode_count'] += 1
        
        # 计算滑动平均
        recent_results = list(self.performance_history[scenario_type])
        if recent_results:
            success_rate = np.mean([r.get('success', 0) for r in recent_results])
            collision_rate = np.mean([r.get('collision', 0) for r in recent_results])
            avg_reward = np.mean([r.get('total_reward', 0) for r in recent_results])
            avg_steps = np.mean([r.get('steps', 0) for r in recent_results])
            
            stats['success_rate'] = success_rate
            stats['collision_rate'] = collision_rate
            stats['avg_reward'] = avg_reward
            stats['avg_steps'] = avg_steps
    
    def get_performance_summary(self) -> Dict[str, Dict]:
        """
        获取性能摘要
        
        Returns:
            performance_summary: 性能摘要
        """
        return dict(self.scenario_stats)
    
    def get_weak_scenarios(self, success_threshold: float = 0.7) -> List[str]:
        """
        获取表现较弱的场景
        
        Args:
            success_threshold: 成功率阈值
            
        Returns:
            weak_scenarios: 表现较弱的场景列表
        """
        weak_scenarios = []
        for scenario_type, stats in self.scenario_stats.items():
            if stats['success_rate'] < success_threshold and stats['episode_count'] >= 10:
                weak_scenarios.append(scenario_type)
        return weak_scenarios


class AdaptiveDifficultyManager:
    """
    自适应难度管理器
    """
    
    def __init__(self, base_difficulty: float = 1.0, 
                 difficulty_range: Tuple[float, float] = (0.5, 2.0),
                 adjustment_rate: float = 0.1):
        self.base_difficulty = base_difficulty
        self.difficulty_range = difficulty_range
        self.adjustment_rate = adjustment_rate
        self.scenario_difficulties = defaultdict(lambda: base_difficulty)
        
    def adjust_difficulty(self, scenario_type: str, performance_metrics: Dict) -> float:
        """
        根据性能调整场景难度
        
        Args:
            scenario_type: 场景类型
            performance_metrics: 性能指标
            
        Returns:
            new_difficulty: 新的难度系数
        """
        current_difficulty = self.scenario_difficulties[scenario_type]
        success_rate = performance_metrics.get('success_rate', 0.5)
        collision_rate = performance_metrics.get('collision_rate', 0.1)
        
        # 基于成功率和碰撞率调整难度
        if success_rate > 0.8 and collision_rate < 0.05:
            # 表现很好，增加难度
            difficulty_adjustment = self.adjustment_rate
        elif success_rate < 0.6 or collision_rate > 0.15:
            # 表现较差，降低难度
            difficulty_adjustment = -self.adjustment_rate
        else:
            # 表现一般，保持当前难度
            difficulty_adjustment = 0.0
        
        new_difficulty = current_difficulty + difficulty_adjustment
        new_difficulty = np.clip(new_difficulty, *self.difficulty_range)
        
        self.scenario_difficulties[scenario_type] = new_difficulty
        return new_difficulty
    
    def get_difficulty(self, scenario_type: str) -> float:
        """
        获取场景难度
        
        Args:
            scenario_type: 场景类型
            
        Returns:
            difficulty: 难度系数
        """
        return self.scenario_difficulties[scenario_type]


class CurriculumManager:
    """
    课程学习管理器
    """
    
    def __init__(self, 
                 success_threshold: float = 0.7,
                 collision_threshold: float = 0.1,
                 performance_window: int = 100,
                 min_episodes_per_scenario: int = 10):
        
        self.success_threshold = success_threshold
        self.collision_threshold = collision_threshold
        self.performance_window = performance_window
        self.min_episodes_per_scenario = min_episodes_per_scenario
        
        # 初始化组件
        self.performance_tracker = ScenarioPerformanceTracker(performance_window)
        self.difficulty_manager = AdaptiveDifficultyManager()
        
        # 场景配置
        self.scenario_types = [
            'straight_path',      # 直线路径
            'curved_path',        # 弯道
            'static_obstacles',   # 静态障碍物
            'dynamic_obstacles',  # 动态障碍物
            'intersection',       # 交叉路口
            'narrow_passage',     # 窄道
            'complex_scenario'    # 复杂场景
        ]
        
        # 初始场景权重
        self.scenario_weights = {scenario: 1.0 for scenario in self.scenario_types}
        self.scenario_episode_counts = defaultdict(int)
        
        # 训练阶段
        self.training_phase = 'basic'  # basic, intermediate, advanced
        self.phase_transition_threshold = 0.8
        
    def update_curriculum(self, scenario_type: str, episode_result: Dict):
        """
        更新课程学习状态
        
        Args:
            scenario_type: 场景类型
            episode_result: 回合结果
        """
        # 更新性能跟踪
        self.performance_tracker.update_performance(scenario_type, episode_result)
        self.scenario_episode_counts[scenario_type] += 1
        
        # 调整场景难度
        performance_metrics = self.performance_tracker.scenario_stats[scenario_type]
        new_difficulty = self.difficulty_manager.adjust_difficulty(scenario_type, performance_metrics)
        
        # 更新场景权重
        self._update_scenario_weights()
        
        # 检查阶段转换
        self._check_phase_transition()
    
    def _update_scenario_weights(self):
        """
        更新场景权重
        """
        performance_summary = self.performance_tracker.get_performance_summary()
        weak_scenarios = self.performance_tracker.get_weak_scenarios(self.success_threshold)
        
        # 为表现较弱的场景增加权重
        for scenario_type in self.scenario_types:
            if scenario_type in weak_scenarios:
                # 增加权重，但不超过2.0
                self.scenario_weights[scenario_type] = min(2.0, 
                    self.scenario_weights[scenario_type] * 1.1)
            else:
                # 逐渐减少权重，但不低于0.5
                self.scenario_weights[scenario_type] = max(0.5, 
                    self.scenario_weights[scenario_type] * 0.99)
    
    def _check_phase_transition(self):
        """
        检查训练阶段转换
        """
        performance_summary = self.performance_tracker.get_performance_summary()
        
        if self.training_phase == 'basic':
            # 检查是否可以进入中级阶段
            basic_scenarios = ['straight_path', 'curved_path', 'static_obstacles']
            basic_success_rates = [performance_summary.get(scenario, {}).get('success_rate', 0) 
                                 for scenario in basic_scenarios]
            
            if all(rate >= self.phase_transition_threshold for rate in basic_success_rates):
                self.training_phase = 'intermediate'
                print(f"训练阶段转换: basic -> intermediate")
                
        elif self.training_phase == 'intermediate':
            # 检查是否可以进入高级阶段
            intermediate_scenarios = ['dynamic_obstacles', 'intersection']
            intermediate_success_rates = [performance_summary.get(scenario, {}).get('success_rate', 0) 
                                        for scenario in intermediate_scenarios]
            
            if all(rate >= self.phase_transition_threshold for rate in intermediate_success_rates):
                self.training_phase = 'advanced'
                print(f"训练阶段转换: intermediate -> advanced")
    
    def select_scenario(self) -> Tuple[str, float]:
        """
        选择下一个训练场景
        
        Returns:
            scenario_type: 场景类型
            difficulty: 难度系数
        """
        # 根据当前阶段和权重选择场景
        if self.training_phase == 'basic':
            available_scenarios = ['straight_path', 'curved_path', 'static_obstacles']
        elif self.training_phase == 'intermediate':
            available_scenarios = ['straight_path', 'curved_path', 'static_obstacles', 
                                 'dynamic_obstacles', 'intersection']
        else:  # advanced
            available_scenarios = self.scenario_types
        
        # 计算选择概率
        weights = [self.scenario_weights[scenario] for scenario in available_scenarios]
        probabilities = np.array(weights) / np.sum(weights)
        
        # 选择场景
        scenario_type = np.random.choice(available_scenarios, p=probabilities)
        difficulty = self.difficulty_manager.get_difficulty(scenario_type)
        
        return scenario_type, difficulty
    
    def get_training_statistics(self) -> Dict:
        """
        获取训练统计信息
        
        Returns:
            statistics: 训练统计信息
        """
        performance_summary = self.performance_tracker.get_performance_summary()
        
        return {
            'training_phase': self.training_phase,
            'scenario_weights': dict(self.scenario_weights),
            'scenario_difficulties': dict(self.difficulty_manager.scenario_difficulties),
            'performance_summary': performance_summary,
            'episode_counts': dict(self.scenario_episode_counts),
            'weak_scenarios': self.performance_tracker.get_weak_scenarios(self.success_threshold)
        }
    
    def save_curriculum_state(self, filepath: str):
        """
        保存课程学习状态
        
        Args:
            filepath: 保存路径
        """
        state = {
            'training_phase': self.training_phase,
            'scenario_weights': dict(self.scenario_weights),
            'scenario_difficulties': dict(self.difficulty_manager.scenario_difficulties),
            'scenario_episode_counts': dict(self.scenario_episode_counts),
            'performance_summary': self.performance_tracker.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_curriculum_state(self, filepath: str):
        """
        加载课程学习状态
        
        Args:
            filepath: 加载路径
        """
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.training_phase = state.get('training_phase', 'basic')
        self.scenario_weights = state.get('scenario_weights', {scenario: 1.0 for scenario in self.scenario_types})
        self.difficulty_manager.scenario_difficulties = defaultdict(lambda: 1.0, 
                                                                  state.get('scenario_difficulties', {}))
        self.scenario_episode_counts = defaultdict(int, state.get('scenario_episode_counts', {}))


class ScenarioGenerator:
    """
    场景生成器 - 根据课程学习需求生成场景
    """
    
    def __init__(self, curriculum_manager: CurriculumManager):
        self.curriculum_manager = curriculum_manager
        
    def generate_scenario_config(self, scenario_type: str, difficulty: float) -> Dict:
        """
        生成场景配置
        
        Args:
            scenario_type: 场景类型
            difficulty: 难度系数
            
        Returns:
            scenario_config: 场景配置
        """
        base_config = {
            'scenario_type': scenario_type,
            'difficulty': difficulty,
            'max_steps': 1000,
            'timeout': 300
        }
        
        if scenario_type == 'straight_path':
            base_config.update({
                'path_length': int(100 * difficulty),
                'obstacle_count': 0,
                'dynamic_obstacles': 0
            })
        elif scenario_type == 'curved_path':
            base_config.update({
                'path_length': int(150 * difficulty),
                'curvature': 0.1 * difficulty,
                'obstacle_count': max(0, int(2 * difficulty - 1))
            })
        elif scenario_type == 'static_obstacles':
            base_config.update({
                'obstacle_count': int(3 * difficulty),
                'obstacle_density': 0.1 * difficulty
            })
        elif scenario_type == 'dynamic_obstacles':
            base_config.update({
                'dynamic_obstacles': int(2 * difficulty),
                'obstacle_speed': 2.0 * difficulty,
                'obstacle_behavior': 'random' if difficulty > 1.5 else 'predictable'
            })
        elif scenario_type == 'intersection':
            base_config.update({
                'intersection_type': 'cross' if difficulty > 1.0 else 't_junction',
                'traffic_density': 0.3 * difficulty,
                'traffic_speed': 3.0 * difficulty
            })
        elif scenario_type == 'narrow_passage':
            base_config.update({
                'passage_width': max(5.0, 8.0 - 2.0 * difficulty),
                'obstacle_count': int(1 * difficulty)
            })
        elif scenario_type == 'complex_scenario':
            base_config.update({
                'obstacle_count': int(5 * difficulty),
                'dynamic_obstacles': int(3 * difficulty),
                'intersection_count': int(2 * difficulty),
                'narrow_passages': int(1 * difficulty)
            })
        
        return base_config
