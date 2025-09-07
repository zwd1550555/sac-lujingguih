# -*- coding: utf-8 -*-
"""
训练场景可视化工具
用于可视化智能体在不同训练场景中的表现

主要功能:
- 场景可视化: 实时显示训练场景和智能体行为
- 轨迹记录: 记录和回放智能体的运动轨迹
- 性能分析: 可视化性能指标和统计数据
- 交互式调试: 支持暂停、单步执行等调试功能

技术特性:
- 实时渲染: 使用matplotlib进行实时场景渲染
- 多场景支持: 支持不同类型的训练场景
- 数据记录: 完整的训练数据记录和回放
- 性能监控: 实时性能指标显示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.patches as patches
from collections import deque
import time
import threading
from typing import Dict, List, Tuple, Optional
import json
import os


class ScenarioVisualizer:
    """
    训练场景可视化器
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 8),
                 update_interval: int = 50,
                 save_animation: bool = False,
                 animation_path: str = "training_animation.gif"):
        
        self.figsize = figsize
        self.update_interval = update_interval
        self.save_animation = save_animation
        self.animation_path = animation_path
        
        # 初始化图形界面
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(-50, 150)
        self.ax.set_ylim(-50, 100)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('履带车智能体训练场景可视化', fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X坐标 (米)', fontsize=12)
        self.ax.set_ylabel('Y坐标 (米)', fontsize=12)
        
        # 场景元素
        self.ego_vehicle = None
        self.obstacles = []
        self.goal = None
        self.road_boundaries = []
        self.trajectory = deque(maxlen=1000)
        
        # 性能指标
        self.performance_text = None
        self.reward_history = deque(maxlen=100)
        self.success_rate = 0.0
        self.collision_rate = 0.0
        
        # 动画控制
        self.animation = None
        self.is_running = False
        self.is_paused = False
        
        # 数据记录
        self.episode_data = []
        self.current_episode = 0
        
    def setup_scenario(self, scenario_config: Dict):
        """
        设置训练场景
        
        Args:
            scenario_config: 场景配置字典
        """
        self.scenario_config = scenario_config
        self.scenario_type = scenario_config.get('scenario_type', 'straight_path')
        self.difficulty = scenario_config.get('difficulty', 1.0)
        
        # 清除之前的场景元素
        self.clear_scenario()
        
        # 根据场景类型设置场景
        if self.scenario_type == 'straight_path':
            self._setup_straight_path_scenario()
        elif self.scenario_type == 'curved_path':
            self._setup_curved_path_scenario()
        elif self.scenario_type == 'static_obstacles':
            self._setup_static_obstacles_scenario()
        elif self.scenario_type == 'dynamic_obstacles':
            self._setup_dynamic_obstacles_scenario()
        elif self.scenario_type == 'intersection':
            self._setup_intersection_scenario()
        elif self.scenario_type == 'narrow_passage':
            self._setup_narrow_passage_scenario()
        elif self.scenario_type == 'complex_scenario':
            self._setup_complex_scenario()
        
        # 设置性能显示
        self._setup_performance_display()
        
    def _setup_straight_path_scenario(self):
        """设置直线路径场景"""
        # 道路边界
        road_width = 8.0
        road_length = 100.0
        
        # 左边界
        left_boundary = Rectangle((-road_width/2, 0), 0.5, road_length, 
                                facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_patch(left_boundary)
        self.road_boundaries.append(left_boundary)
        
        # 右边界
        right_boundary = Rectangle((road_width/2, 0), 0.5, road_length, 
                                 facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_patch(right_boundary)
        self.road_boundaries.append(right_boundary)
        
        # 目标点
        self.goal = Circle((0, 90), 3, facecolor='green', edgecolor='darkgreen', 
                          linewidth=2, alpha=0.7)
        self.ax.add_patch(self.goal)
        
        # 起始点
        start_point = Circle((0, 0), 2, facecolor='blue', edgecolor='darkblue', 
                           linewidth=2, alpha=0.7)
        self.ax.add_patch(start_point)
        
    def _setup_curved_path_scenario(self):
        """设置弯道场景"""
        # 弯道路径
        curve_radius = 30.0
        curve_center = (40, 30)
        
        # 内弯边界
        inner_curve = Circle(curve_center, curve_radius - 4, 
                           facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_patch(inner_curve)
        self.road_boundaries.append(inner_curve)
        
        # 外弯边界
        outer_curve = Circle(curve_center, curve_radius + 4, 
                           facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_patch(outer_curve)
        self.road_boundaries.append(outer_curve)
        
        # 目标点
        goal_x = curve_center[0] + curve_radius * np.cos(np.pi/4)
        goal_y = curve_center[1] + curve_radius * np.sin(np.pi/4)
        self.goal = Circle((goal_x, goal_y), 3, facecolor='green', 
                          edgecolor='darkgreen', linewidth=2, alpha=0.7)
        self.ax.add_patch(self.goal)
        
        # 起始点
        start_x = curve_center[0] + curve_radius * np.cos(-np.pi/4)
        start_y = curve_center[1] + curve_radius * np.sin(-np.pi/4)
        start_point = Circle((start_x, start_y), 2, facecolor='blue', 
                           edgecolor='darkblue', linewidth=2, alpha=0.7)
        self.ax.add_patch(start_point)
        
    def _setup_static_obstacles_scenario(self):
        """设置静态障碍物场景"""
        # 基础道路
        self._setup_straight_path_scenario()
        
        # 静态障碍物
        obstacle_count = int(3 * self.difficulty)
        for i in range(obstacle_count):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(20, 80)
            obstacle = Rectangle((x-1.5, y-1.5), 3, 3, 
                               facecolor='red', edgecolor='darkred', 
                               linewidth=2, alpha=0.8)
            self.ax.add_patch(obstacle)
            self.obstacles.append(obstacle)
            
    def _setup_dynamic_obstacles_scenario(self):
        """设置动态障碍物场景"""
        # 基础道路
        self._setup_straight_path_scenario()
        
        # 动态障碍物
        obstacle_count = int(2 * self.difficulty)
        for i in range(obstacle_count):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(10, 70)
            obstacle = Circle((x, y), 2, facecolor='orange', 
                            edgecolor='darkorange', linewidth=2, alpha=0.8)
            self.ax.add_patch(obstacle)
            self.obstacles.append(obstacle)
            
    def _setup_intersection_scenario(self):
        """设置交叉路口场景"""
        # 主道路
        main_road = Rectangle((-4, -10), 8, 120, 
                            facecolor='lightgray', edgecolor='black', linewidth=2)
        self.ax.add_patch(main_road)
        
        # 交叉道路
        cross_road = Rectangle((-20, 40), 40, 8, 
                             facecolor='lightgray', edgecolor='black', linewidth=2)
        self.ax.add_patch(cross_road)
        
        # 目标点
        self.goal = Circle((0, 100), 3, facecolor='green', 
                          edgecolor='darkgreen', linewidth=2, alpha=0.7)
        self.ax.add_patch(self.goal)
        
        # 起始点
        start_point = Circle((0, -5), 2, facecolor='blue', 
                           edgecolor='darkblue', linewidth=2, alpha=0.7)
        self.ax.add_patch(start_point)
        
        # 交通车辆
        traffic_count = int(3 * self.difficulty)
        for i in range(traffic_count):
            x = np.random.uniform(-15, 15)
            y = np.random.uniform(35, 45)
            vehicle = Rectangle((x-1.5, y-1.5), 3, 3, 
                              facecolor='yellow', edgecolor='darkyellow', 
                              linewidth=2, alpha=0.8)
            self.ax.add_patch(vehicle)
            self.obstacles.append(vehicle)
            
    def _setup_narrow_passage_scenario(self):
        """设置窄道场景"""
        # 窄道边界
        passage_width = max(4.0, 8.0 - 2.0 * self.difficulty)
        
        # 左边界
        left_boundary = Rectangle((-passage_width/2, 0), 0.5, 100, 
                                facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_patch(left_boundary)
        self.road_boundaries.append(left_boundary)
        
        # 右边界
        right_boundary = Rectangle((passage_width/2, 0), 0.5, 100, 
                                 facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_patch(right_boundary)
        self.road_boundaries.append(right_boundary)
        
        # 目标点
        self.goal = Circle((0, 90), 2, facecolor='green', 
                          edgecolor='darkgreen', linewidth=2, alpha=0.7)
        self.ax.add_patch(self.goal)
        
        # 起始点
        start_point = Circle((0, 0), 1.5, facecolor='blue', 
                           edgecolor='darkblue', linewidth=2, alpha=0.7)
        self.ax.add_patch(start_point)
        
    def _setup_complex_scenario(self):
        """设置复杂场景"""
        # 组合多个场景元素
        self._setup_curved_path_scenario()
        
        # 添加更多障碍物
        obstacle_count = int(5 * self.difficulty)
        for i in range(obstacle_count):
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(10, 80)
            obstacle = Rectangle((x-1, y-1), 2, 2, 
                               facecolor='purple', edgecolor='darkpurple', 
                               linewidth=2, alpha=0.8)
            self.ax.add_patch(obstacle)
            self.obstacles.append(obstacle)
            
    def _setup_performance_display(self):
        """设置性能显示"""
        # 性能文本
        self.performance_text = self.ax.text(0.02, 0.98, '', 
                                           transform=self.ax.transAxes, 
                                           verticalalignment='top',
                                           bbox=dict(boxstyle='round', 
                                                   facecolor='white', 
                                                   alpha=0.8),
                                           fontsize=10)
        
    def update_ego_vehicle(self, x: float, y: float, yaw: float, 
                          v: float = 0.0, action: np.ndarray = None):
        """
        更新自车位置和状态
        
        Args:
            x: X坐标
            y: Y坐标
            yaw: 航向角
            v: 速度
            action: 当前动作
        """
        # 移除之前的自车
        if self.ego_vehicle is not None:
            self.ego_vehicle.remove()
        
        # 创建新的自车
        vehicle_length = 4.0
        vehicle_width = 2.0
        
        # 计算车辆四个角点
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 车辆中心到四个角的向量
        corners = np.array([
            [-vehicle_length/2, -vehicle_width/2],
            [vehicle_length/2, -vehicle_width/2],
            [vehicle_length/2, vehicle_width/2],
            [-vehicle_length/2, vehicle_width/2]
        ])
        
        # 旋转和平移
        rotated_corners = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]]) @ corners.T
        rotated_corners = rotated_corners.T + np.array([x, y])
        
        # 创建车辆多边形
        self.ego_vehicle = Polygon(rotated_corners, 
                                 facecolor='blue', edgecolor='darkblue', 
                                 linewidth=2, alpha=0.8)
        self.ax.add_patch(self.ego_vehicle)
        
        # 添加速度箭头
        if v > 0.1:
            arrow_length = v * 2  # 箭头长度与速度成正比
            arrow_end_x = x + arrow_length * cos_yaw
            arrow_end_y = y + arrow_length * sin_yaw
            
            self.ax.annotate('', xy=(arrow_end_x, arrow_end_y), 
                           xytext=(x, y),
                           arrowprops=dict(arrowstyle='->', 
                                         color='red', lw=2))
        
        # 记录轨迹
        self.trajectory.append((x, y))
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            trajectory_x = [point[0] for point in self.trajectory]
            trajectory_y = [point[1] for point in self.trajectory]
            self.ax.plot(trajectory_x, trajectory_y, 'b-', alpha=0.5, linewidth=1)
        
    def update_obstacles(self, obstacle_positions: List[Tuple[float, float, float]]):
        """
        更新障碍物位置
        
        Args:
            obstacle_positions: 障碍物位置列表 [(x, y, yaw), ...]
        """
        # 清除之前的障碍物
        for obstacle in self.obstacles:
            obstacle.remove()
        self.obstacles.clear()
        
        # 添加新的障碍物
        for i, (x, y, yaw) in enumerate(obstacle_positions):
            if self.scenario_type == 'dynamic_obstacles':
                obstacle = Circle((x, y), 2, facecolor='orange', 
                                edgecolor='darkorange', linewidth=2, alpha=0.8)
            else:
                obstacle = Rectangle((x-1.5, y-1.5), 3, 3, 
                                   facecolor='red', edgecolor='darkred', 
                                   linewidth=2, alpha=0.8)
            self.ax.add_patch(obstacle)
            self.obstacles.append(obstacle)
            
    def update_performance(self, episode_reward: float, success: bool, 
                          collision: bool, steps: int):
        """
        更新性能指标
        
        Args:
            episode_reward: 回合奖励
            success: 是否成功
            collision: 是否碰撞
            steps: 步数
        """
        self.reward_history.append(episode_reward)
        
        # 更新成功率
        if success:
            self.success_rate = min(1.0, self.success_rate + 0.01)
        else:
            self.success_rate = max(0.0, self.success_rate - 0.005)
            
        # 更新碰撞率
        if collision:
            self.collision_rate = min(1.0, self.collision_rate + 0.01)
        else:
            self.collision_rate = max(0.0, self.collision_rate - 0.005)
        
        # 更新性能显示
        if self.performance_text is not None:
            avg_reward = np.mean(self.reward_history) if self.reward_history else 0
            performance_str = f"""
场景类型: {self.scenario_type}
难度等级: {self.difficulty:.2f}
当前奖励: {episode_reward:.2f}
平均奖励: {avg_reward:.2f}
成功率: {self.success_rate:.2%}
碰撞率: {self.collision_rate:.2%}
步数: {steps}
状态: {'成功' if success else '碰撞' if collision else '进行中'}
            """.strip()
            
            self.performance_text.set_text(performance_str)
            
    def clear_scenario(self):
        """清除场景元素"""
        # 清除自车
        if self.ego_vehicle is not None:
            self.ego_vehicle.remove()
            self.ego_vehicle = None
            
        # 清除障碍物
        for obstacle in self.obstacles:
            obstacle.remove()
        self.obstacles.clear()
        
        # 清除目标
        if self.goal is not None:
            self.goal.remove()
            self.goal = None
            
        # 清除道路边界
        for boundary in self.road_boundaries:
            boundary.remove()
        self.road_boundaries.clear()
        
        # 清除轨迹
        self.trajectory.clear()
        
    def start_animation(self):
        """启动动画"""
        if not self.is_running:
            self.is_running = True
            self.animation = animation.FuncAnimation(
                self.fig, self._animate, interval=self.update_interval, 
                blit=False, repeat=True
            )
            plt.show()
            
    def _animate(self, frame):
        """动画更新函数"""
        if not self.is_paused:
            # 这里可以添加实时数据更新逻辑
            pass
            
    def pause_animation(self):
        """暂停动画"""
        self.is_paused = not self.is_paused
        
    def stop_animation(self):
        """停止动画"""
        self.is_running = False
        if self.animation is not None:
            self.animation.event_source.stop()
            
    def save_episode_data(self, episode_data: Dict):
        """
        保存回合数据
        
        Args:
            episode_data: 回合数据字典
        """
        self.episode_data.append(episode_data)
        
        # 保存到文件
        if len(self.episode_data) % 10 == 0:  # 每10个回合保存一次
            self._save_data_to_file()
            
    def _save_data_to_file(self):
        """保存数据到文件"""
        data_file = f"visualization_data_episode_{self.current_episode}.json"
        with open(data_file, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
            
    def load_episode_data(self, data_file: str):
        """
        加载回合数据
        
        Args:
            data_file: 数据文件路径
        """
        with open(data_file, 'r') as f:
            self.episode_data = json.load(f)
            
    def replay_episode(self, episode_index: int = -1):
        """
        回放指定回合
        
        Args:
            episode_index: 回合索引，-1表示最后一个回合
        """
        if not self.episode_data:
            print("没有可回放的数据")
            return
            
        episode = self.episode_data[episode_index]
        
        # 设置场景
        self.setup_scenario(episode['scenario_config'])
        
        # 回放轨迹
        for step_data in episode['steps']:
            self.update_ego_vehicle(
                step_data['ego_x'], 
                step_data['ego_y'], 
                step_data['ego_yaw'],
                step_data.get('ego_v', 0.0),
                step_data.get('action')
            )
            
            if 'obstacle_positions' in step_data:
                self.update_obstacles(step_data['obstacle_positions'])
                
            self.update_performance(
                step_data.get('reward', 0.0),
                step_data.get('success', False),
                step_data.get('collision', False),
                step_data.get('step', 0)
            )
            
            plt.pause(0.1)  # 暂停0.1秒
            
    def show_statistics(self):
        """显示统计信息"""
        if not self.episode_data:
            print("没有统计数据")
            return
            
        # 计算统计信息
        total_episodes = len(self.episode_data)
        success_count = sum(1 for ep in self.episode_data if ep.get('success', False))
        collision_count = sum(1 for ep in self.episode_data if ep.get('collision', False))
        
        avg_reward = np.mean([ep.get('total_reward', 0) for ep in self.episode_data])
        avg_steps = np.mean([ep.get('steps', 0) for ep in self.episode_data])
        
        print(f"""
=== 训练统计信息 ===
总回合数: {total_episodes}
成功率: {success_count/total_episodes:.2%}
碰撞率: {collision_count/total_episodes:.2%}
平均奖励: {avg_reward:.2f}
平均步数: {avg_steps:.1f}
        """.strip())


class TrainingVisualizer:
    """
    训练过程可视化器
    """
    
    def __init__(self, visualizer: ScenarioVisualizer):
        self.visualizer = visualizer
        self.is_monitoring = False
        
    def start_monitoring(self, env, agent, max_episodes: int = 100):
        """
        开始监控训练过程
        
        Args:
            env: 环境对象
            agent: 智能体对象
            max_episodes: 最大监控回合数
        """
        self.is_monitoring = True
        self.env = env
        self.agent = agent
        self.max_episodes = max_episodes
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self._monitor_training)
        monitor_thread.daemon = True
        monitor_thread.start()
        
    def _monitor_training(self):
        """监控训练过程"""
        episode = 0
        
        while self.is_monitoring and episode < self.max_episodes:
            # 运行一个回合
            episode_data = self._run_episode(episode)
            
            # 更新可视化
            self.visualizer.save_episode_data(episode_data)
            
            episode += 1
            
    def _run_episode(self, episode: int) -> Dict:
        """运行一个回合"""
        # 重置环境
        observation = self.env.reset()
        
        # 设置场景
        scenario_config = {
            'scenario_type': 'straight_path',  # 可以从环境获取
            'difficulty': 1.0
        }
        self.visualizer.setup_scenario(scenario_config)
        
        episode_data = {
            'episode': episode,
            'scenario_config': scenario_config,
            'steps': [],
            'total_reward': 0.0,
            'success': False,
            'collision': False,
            'steps': 0
        }
        
        step = 0
        done = False
        
        while not done and step < 1000:
            # 获取状态和动作
            state = self._extract_state(observation)
            action = self.agent.select_action(state)
            
            # 执行动作
            next_observation = self.env.step_rl(action)
            
            # 记录数据
            step_data = {
                'step': step,
                'ego_x': observation['vehicle_info']['ego']['x'],
                'ego_y': observation['vehicle_info']['ego']['y'],
                'ego_yaw': observation['vehicle_info']['ego']['yaw_rad'],
                'ego_v': observation['vehicle_info']['ego']['v_mps'],
                'action': action,
                'reward': 0.0,  # 可以从环境获取
                'success': next_observation['test_setting']['end'] == 0,
                'collision': next_observation['test_setting']['end'] == 1
            }
            
            episode_data['steps'].append(step_data)
            episode_data['total_reward'] += step_data['reward']
            
            # 更新可视化
            self.visualizer.update_ego_vehicle(
                step_data['ego_x'], 
                step_data['ego_y'], 
                step_data['ego_yaw'],
                step_data['ego_v'],
                action
            )
            
            self.visualizer.update_performance(
                episode_data['total_reward'],
                step_data['success'],
                step_data['collision'],
                step
            )
            
            # 检查终止条件
            done = next_observation['test_setting']['end'] != -1
            observation = next_observation
            step += 1
            
        episode_data['success'] = next_observation['test_setting']['end'] == 0
        episode_data['collision'] = next_observation['test_setting']['end'] == 1
        episode_data['steps'] = step
        
        return episode_data
        
    def _extract_state(self, observation: Dict) -> np.ndarray:
        """提取状态（简化版本）"""
        ego = observation['vehicle_info']['ego']
        goal = observation['test_setting']['goal']
        
        # 简化的状态提取
        state = [
            float(ego['v_mps']) / 10.0,
            float(ego['yawrate_radps']) / 2.0,
            float(ego['x']),
            float(ego['y']),
            float(ego['yaw_rad'])
        ]
        
        return np.array(state, dtype=np.float32)
        
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False


def main():
    """主函数 - 演示可视化功能"""
    # 创建可视化器
    visualizer = ScenarioVisualizer()
    
    # 设置测试场景
    scenario_config = {
        'scenario_type': 'intersection',
        'difficulty': 1.5
    }
    
    visualizer.setup_scenario(scenario_config)
    
    # 模拟智能体运动
    for i in range(100):
        x = i * 0.5
        y = 10 + i * 0.3
        yaw = i * 0.01
        
        visualizer.update_ego_vehicle(x, y, yaw, v=2.0)
        visualizer.update_performance(
            episode_reward=100 - i,
            success=False,
            collision=False,
            steps=i
        )
        
        plt.pause(0.1)
        
    # 显示最终结果
    visualizer.update_performance(
        episode_reward=50,
        success=True,
        collision=False,
        steps=100
    )
    
    plt.show()


if __name__ == '__main__':
    main()
