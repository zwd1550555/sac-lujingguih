# -*- coding: utf-8 -*-
"""
高级训练脚本 - 集成所有优化技术
实现下一代智能体的完整训练流程

主要功能:
- 集成因果推理和主动决策
- 概率世界模型训练
- 序列化BPTT训练
- 自动化课程学习
- 高级奖励设计

技术特性:
- 四支柱优化: 因果推理、网络效能、训练策略、模型鲁棒性
- 自适应训练: 根据性能自动调整训练策略
- 多维度评估: 综合评估智能体性能
- 智能监控: 实时监控训练过程
"""

import os
import sys
import math
import random
import argparse
import numpy as np
import torch
import yaml
from datetime import datetime
from collections import deque

# 导入项目模块
from dynamic_scenes.env import Env
from dynamic_scenes.scenarioOrganizer1 import ScenarioOrganizer
from dynamic_scenes.lookup import CollisionLookup
from dynamic_scenes.observation import Observation

# 导入高级组件
from agent_advanced import AdvancedLiquidSACAgent, InterventionBasedDecisionMaker, ProbabilisticCausalReasoner, MCTS_Config
from probabilistic_trainer import ProbabilisticWorldModelTrainer, UncertaintyAwareRewardShaping
from sequence_trainer import SequenceBasedSACAgent, BPTTTrainer
from curriculum_manager import CurriculumManager, ScenarioGenerator
from advanced_reward import AdvancedRewardCalculator, RewardVisualizer
from mcts_planner import MCTS_Planner

# 导入原有组件
from agent import SACAgent
from train_rl import extract_state, apply_safety_constraints

# TensorBoard支持
try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False


class AdvancedTrainingManager:
    """
    高级训练管理器
    """
    
    def __init__(self, config_path: str, tag: str = "advanced_training"):
        self.config_path = config_path
        self.tag = tag
        self.config = self._load_config()
        
        # 初始化组件
        self._initialize_components()
        
        # 训练状态
        self.episode = 0
        self.total_steps = 0
        self.best_performance = -float('inf')
        
        # 性能记录
        self.performance_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=1000)
        
    def _load_config(self) -> dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 环境组件
        self.env = Env()
        self.scenario_organizer = ScenarioOrganizer()
        self.collision_lookup = CollisionLookup()
        self.observation = Observation()
        
        # 智能体组件
        state_dim = 5 + 4 * self.config['train']['num_obstacles'] + 5  # 22维状态
        action_dim = 2
        
        # MCTS配置
        mcts_config = MCTS_Config(
            num_simulations=self.config.get('mcts', {}).get('num_simulations', 100),
            exploration_constant=self.config.get('mcts', {}).get('exploration_constant', 1.5),
            gamma=self.config.get('mcts', {}).get('gamma', 0.99),
            max_depth=self.config.get('mcts', {}).get('max_depth', 20),
            temperature=self.config.get('mcts', {}).get('temperature', 1.0)
        )
        
        self.agent = AdvancedLiquidSACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=self.config['network']['hidden_dim'],
            replay_buffer_capacity=self.config['train']['replay_buffer_capacity'],
            actor_lr=self.config['optimizer']['actor_lr'],
            critic_lr=self.config['optimizer']['critic_lr'],
            gamma=self.config['train']['gamma'],
            tau=self.config['train']['tau'],
            alpha=self.config['train']['alpha'],
            sequence_length=self.config['train']['sequence_length'],
            use_intervention_decision=self.config.get('causal', {}).get('enable_intervention', True),
            use_mcts_planning=self.config.get('mcts', {}).get('enable_mcts', False),
            mcts_config=mcts_config
        )
        
        # 世界模型和因果推理器
        from agent_advanced import ProbabilisticWorldModel
        self.world_model = ProbabilisticWorldModel(state_dim, action_dim, 
                                                 self.config['network']['hidden_dim'])
        self.causal_reasoner = ProbabilisticCausalReasoner(self.world_model, self.agent.device)
        
        # 决策制定器
        self.decision_maker = InterventionBasedDecisionMaker(
            self.causal_reasoner,
            reward_weight=self.config['causal']['reward_weight'],
            safety_weight=self.config['causal']['safety_weight'],
            uncertainty_weight=self.config['causal']['uncertainty_weight'],
            prediction_horizon=self.config['causal']['prediction_horizon']
        )
        
        self.agent.set_decision_maker(self.decision_maker)
        
        # 设置MCTS规划器
        self.agent.set_mcts_planner(self.world_model, self.agent.critic)
        
        # 训练器
        self.world_model_trainer = ProbabilisticWorldModelTrainer(
            self.world_model,
            lr=self.config['causal']['world_model_lr']
        )
        
        # 课程学习管理器
        self.curriculum_manager = CurriculumManager(
            success_threshold=self.config['curriculum']['success_threshold'],
            collision_threshold=self.config['curriculum']['collision_threshold']
        )
        
        # 场景生成器
        self.scenario_generator = ScenarioGenerator(self.curriculum_manager)
        
        # 奖励计算器
        self.reward_calculator = AdvancedRewardCalculator(
            comfort_weight=self.config['reward']['comfort_weight'],
            efficiency_weight=self.config['reward']['efficiency_weight'],
            safety_weight=self.config['reward']['safety_weight'],
            goal_weight=self.config['reward']['goal_weight']
        )
        
        # 奖励可视化器
        self.reward_visualizer = RewardVisualizer()
        
        # TensorBoard
        if TB_AVAILABLE:
            self.writer = SummaryWriter(f'runs/{self.tag}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        else:
            self.writer = None
    
    def train_episode(self) -> dict:
        """训练一个episode"""
        # 选择场景
        scenario_type, difficulty = self.curriculum_manager.select_scenario()
        scenario_config = self.scenario_generator.generate_scenario_config(scenario_type, difficulty)
        
        # 重置环境
        observation = self.env.reset()
        state = extract_state(observation, self.collision_lookup, 
                            self.config['train']['num_obstacles'])
        
        # 初始化隐藏状态
        actor_hidden = None
        
        # Episode统计
        episode_reward = 0
        episode_steps = 0
        episode_components = {
            'comfort_reward': 0,
            'efficiency_reward': 0,
            'safety_reward': 0,
            'goal_reward': 0
        }
        
        # 训练循环
        for step in range(self.config['train']['max_steps']):
            # 选择动作
            action, actor_hidden = self.agent.select_action(state, actor_hidden)
            
            # 应用安全约束
            action = apply_safety_constraints(action, self.env, observation)
            
            # 执行动作
            next_obs = self.env.step_rl(action)
            next_state = extract_state(next_obs, self.collision_lookup, 
                                     self.config['train']['num_obstacles'])
            
            # 计算奖励
            reward, reward_components = self.reward_calculator.calculate_advanced_reward(
                next_obs, observation, action, self.config['reward']
            )
            
            # 记录奖励组件
            for component, value in reward_components.items():
                if component in episode_components:
                    episode_components[component] += value
            
            # 存储经验
            done = next_obs['test_setting']['end'] != -1
            self.agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新状态
            state = next_state
            observation = next_obs
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            
            # 训练智能体
            if len(self.agent.replay_buffer) > self.config['train']['warmup_steps']:
                if self.total_steps % self.config['train']['update_interval'] == 0:
                    metrics = self.agent.update(self.config['train']['batch_size'])
                    if metrics and self.writer:
                        self._log_training_metrics(metrics)
            
            # 训练世界模型
            if self.total_steps % self.config['causal']['world_model_update_interval'] == 0:
                world_model_metrics = self.world_model_trainer.train_from_buffer(
                    self.agent.replay_buffer, 
                    self.config['train']['batch_size'],
                    epochs=1
                )
                if world_model_metrics and self.writer:
                    self._log_world_model_metrics(world_model_metrics)
            
            if done:
                break
        
        # 更新课程学习
        episode_result = {
            'success': next_obs['test_setting']['end'] == 0,
            'collision': next_obs['test_setting']['end'] == 1,
            'total_reward': episode_reward,
            'steps': episode_steps,
            'scenario_type': scenario_type,
            'difficulty': difficulty
        }
        
        self.curriculum_manager.update_curriculum(scenario_type, episode_result)
        
        # 记录性能
        self.performance_history.append(episode_result)
        self.reward_history.append(episode_reward)
        
        # 更新奖励可视化器
        self.reward_visualizer.update(reward_components)
        
        return {
            'episode': self.episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'success': episode_result['success'],
            'collision': episode_result['collision'],
            'scenario_type': scenario_type,
            'difficulty': difficulty,
            'components': episode_components
        }
    
    def _log_training_metrics(self, metrics: dict):
        """记录训练指标"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'Training/{key}', value, self.total_steps)
    
    def _log_world_model_metrics(self, metrics: dict):
        """记录世界模型指标"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'WorldModel/{key}', value, self.total_steps)
    
    def train(self):
        """主训练循环"""
        print("=" * 80)
        print("开始高级智能体训练")
        print("=" * 80)
        print(f"配置: {self.config_path}")
        print(f"标签: {self.tag}")
        print(f"最大episodes: {self.config['train']['max_episodes']}")
        print(f"最大步数: {self.config['train']['max_steps']}")
        print(f"批次大小: {self.config['train']['batch_size']}")
        print("=" * 80)
        
        try:
            for episode in range(self.config['train']['max_episodes']):
                self.episode = episode
                
                # 训练一个episode
                episode_info = self.train_episode()
                
                # 打印进度
                if episode % 10 == 0:
                    self._print_progress(episode_info)
                
                        # 记录到TensorBoard
        if self.writer:
            self._log_episode_metrics(episode_info)
            
            # 记录MCTS统计信息
            if self.agent.use_mcts_planning:
                mcts_stats = self.agent.get_mcts_statistics()
                if mcts_stats:
                    for key, value in mcts_stats.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'MCTS/{key}', value, self.episode)
                
                # 保存模型
                if episode % 100 == 0 and episode > 0:
                    self._save_models(episode)
                
                # 更新奖励权重
                if episode % 200 == 0:
                    training_phase = self.curriculum_manager.training_phase
                    self.reward_calculator.update_weights(training_phase)
        
        except KeyboardInterrupt:
            print("\n训练被用户中断")
        
        finally:
            # 保存最终模型
            self._save_models(self.episode, final=True)
            
            # 保存训练状态
            self._save_training_state()
            
            if self.writer:
                self.writer.close()
            
            print("训练完成!")
    
    def _print_progress(self, episode_info: dict):
        """打印训练进度"""
        avg_reward = np.mean(list(self.reward_history)[-10:]) if self.reward_history else 0
        success_rate = np.mean([p['success'] for p in list(self.performance_history)[-10:]]) if self.performance_history else 0
        
        print(f"Episode {episode_info['episode']:4d} | "
              f"Reward: {episode_info['reward']:8.2f} | "
              f"Avg Reward: {avg_reward:8.2f} | "
              f"Success Rate: {success_rate:.2%} | "
              f"Scenario: {episode_info['scenario_type']} | "
              f"Difficulty: {episode_info['difficulty']:.2f}")
    
    def _log_episode_metrics(self, episode_info: dict):
        """记录episode指标"""
        self.writer.add_scalar('Episode/Reward', episode_info['reward'], episode_info['episode'])
        self.writer.add_scalar('Episode/Steps', episode_info['steps'], episode_info['episode'])
        self.writer.add_scalar('Episode/Success', episode_info['success'], episode_info['episode'])
        self.writer.add_scalar('Episode/Collision', episode_info['collision'], episode_info['episode'])
        
        # 记录奖励组件
        for component, value in episode_info['components'].items():
            self.writer.add_scalar(f'Reward/{component}', value, episode_info['episode'])
        
        # 记录课程学习状态
        curriculum_stats = self.curriculum_manager.get_training_statistics()
        self.writer.add_scalar('Curriculum/Phase', 
                              {'basic': 0, 'intermediate': 1, 'advanced': 2}[curriculum_stats['training_phase']], 
                              episode_info['episode'])
    
    def _save_models(self, episode: int, final: bool = False):
        """保存模型"""
        suffix = "final" if final else f"ep_{episode}"
        
        # 保存智能体模型
        agent_path = f"demo/demo_outputs/advanced_sac_model"
        os.makedirs(agent_path, exist_ok=True)
        
        torch.save(self.agent.actor.state_dict(), f"{agent_path}/actor_{suffix}.pth")
        torch.save(self.agent.critic.state_dict(), f"{agent_path}/critic_{suffix}.pth")
        
        # 保存世界模型
        world_model_path = f"demo/demo_outputs/world_model"
        os.makedirs(world_model_path, exist_ok=True)
        torch.save(self.world_model.state_dict(), f"{world_model_path}/world_model_{suffix}.pth")
        
        print(f"模型已保存: {suffix}")
    
    def _save_training_state(self):
        """保存训练状态"""
        state_path = f"demo/demo_outputs/training_state"
        os.makedirs(state_path, exist_ok=True)
        
        # 保存课程学习状态
        self.curriculum_manager.save_curriculum_state(f"{state_path}/curriculum_state.json")
        
        # 保存奖励统计
        reward_stats = self.reward_calculator.get_reward_statistics()
        with open(f"{state_path}/reward_statistics.json", 'w') as f:
            import json
            json.dump(reward_stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='高级智能体训练脚本')
    parser.add_argument('--cfg', type=str, default='config_advanced.yaml', 
                       help='配置文件路径')
    parser.add_argument('--tag', type=str, default='advanced_training', 
                       help='训练标签')
    
    args = parser.parse_args()
    
    # 创建训练管理器
    trainer = AdvancedTrainingManager(args.cfg, args.tag)
    
    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()
