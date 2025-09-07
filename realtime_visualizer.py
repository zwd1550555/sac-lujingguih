# -*- coding: utf-8 -*-
"""
实时训练可视化脚本
在训练过程中实时显示智能体表现

使用方法:
python realtime_visualizer.py
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle, Polygon
import time
import threading
from collections import deque


def _setup_chinese_font():
    """自动适配中文字体，避免中文显示为方块。"""
    candidate_fonts = [
        'Noto Sans CJK SC', 'Noto Serif CJK SC',
        'Source Han Sans SC', 'Source Han Serif SC', 'Source Han Sans CN',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Microsoft YaHei', 'SimHei', 'SimSun', 'DengXian',
        'PingFang SC', 'Heiti SC', 'Sarasa Gothic SC', 'LXGW WenKai'
    ]
    available_font_names = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidate_fonts:
        if name in available_font_names:
            chosen = name
            break
    if chosen is not None:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [chosen, 'DejaVu Sans']
    else:
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('警告: 未检测到中文字体，建议安装 Noto Sans CJK 或 WenQuanYi 字体以获得正确中文显示。')
    plt.rcParams['axes.unicode_minus'] = False
    try:
        print(f"中文字体设置: {plt.rcParams['font.sans-serif'][0]}")
    except Exception:
        pass


class RealtimeTrainingVisualizer:
    """
    实时训练可视化器
    """
    
    def __init__(self):
        # 设置中文字体（自动适配）
        _setup_chinese_font()
        
        # 创建图形
        self.fig = plt.figure(figsize=(15, 10))
        
        # 创建子图
        self.ax_scenario = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.ax_rewards = plt.subplot2grid((3, 3), (0, 2))
        self.ax_performance = plt.subplot2grid((3, 3), (1, 2))
        self.ax_actions = plt.subplot2grid((3, 3), (2, 0))
        self.ax_statistics = plt.subplot2grid((3, 3), (2, 1))
        self.ax_mcts = plt.subplot2grid((3, 3), (2, 2))
        
        # 初始化场景
        self._setup_scenario_plot()
        self._setup_performance_plots()
        
        # 数据存储
        self.episode_rewards = deque(maxlen=100)
        self.success_rates = deque(maxlen=100)
        self.collision_rates = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.mcts_stats = deque(maxlen=50)
        
        # 当前状态
        self.current_episode = 0
        self.current_scenario = 'straight_path'
        self.ego_position = (0, 0)
        self.ego_yaw = 0
        
        # 场景元素
        self.ego_vehicle = None
        self.obstacles = []
        self.goal = None
        self.trajectory = deque(maxlen=200)
        
    def _setup_scenario_plot(self):
        """设置场景图"""
        self.ax_scenario.set_xlim(-20, 80)
        self.ax_scenario.set_ylim(-20, 60)
        self.ax_scenario.set_aspect('equal')
        self.ax_scenario.grid(True, alpha=0.3)
        self.ax_scenario.set_title('训练场景实时显示', fontsize=14, fontweight='bold')
        self.ax_scenario.set_xlabel('X坐标 (米)')
        self.ax_scenario.set_ylabel('Y坐标 (米)')
        
    def _setup_performance_plots(self):
        """设置性能图表"""
        # 奖励曲线
        self.ax_rewards.set_title('奖励曲线', fontsize=12)
        self.ax_rewards.set_xlabel('回合数')
        self.ax_rewards.set_ylabel('奖励')
        self.ax_rewards.grid(True, alpha=0.3)
        
        # 性能指标
        self.ax_performance.set_title('性能指标', fontsize=12)
        self.ax_performance.set_xlabel('回合数')
        self.ax_performance.set_ylabel('成功率/碰撞率')
        self.ax_performance.grid(True, alpha=0.3)
        
        # 动作分布
        self.ax_actions.set_title('动作分布', fontsize=12)
        self.ax_actions.set_xlabel('左履带速度')
        self.ax_actions.set_ylabel('右履带速度')
        self.ax_actions.grid(True, alpha=0.3)
        
        # 统计信息
        self.ax_statistics.set_title('训练统计', fontsize=12)
        self.ax_statistics.axis('off')
        
        # MCTS统计
        self.ax_mcts.set_title('MCTS统计', fontsize=12)
        self.ax_mcts.set_xlabel('时间步')
        self.ax_mcts.set_ylabel('模拟次数')
        self.ax_mcts.grid(True, alpha=0.3)
        
    def update_scenario(self, scenario_type: str, difficulty: float):
        """更新场景"""
        self.current_scenario = scenario_type
        self.ax_scenario.clear()
        self._setup_scenario_plot()
        
        # 清除之前的元素
        self.obstacles.clear()
        self.trajectory.clear()
        
        # 根据场景类型绘制
        if scenario_type == 'straight_path':
            self._draw_straight_path(difficulty)
        elif scenario_type == 'curved_path':
            self._draw_curved_path(difficulty)
        elif scenario_type == 'intersection':
            self._draw_intersection(difficulty)
        elif scenario_type == 'narrow_passage':
            self._draw_narrow_passage(difficulty)
            
    def _draw_straight_path(self, difficulty: float):
        """绘制直线路径"""
        # 道路边界
        road_width = 8.0
        road_length = 60.0
        
        # 左边界
        left_boundary = Rectangle((-road_width/2, 0), 0.5, road_length, 
                                facecolor='gray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(left_boundary)
        
        # 右边界
        right_boundary = Rectangle((road_width/2, 0), 0.5, road_length, 
                                 facecolor='gray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(right_boundary)
        
        # 目标点
        self.goal = Circle((0, 50), 3, facecolor='green', edgecolor='darkgreen', 
                          linewidth=2, alpha=0.8)
        self.ax_scenario.add_patch(self.goal)
        
        # 起始点
        start_point = Circle((0, 0), 2, facecolor='blue', edgecolor='darkblue', 
                           linewidth=2, alpha=0.8)
        self.ax_scenario.add_patch(start_point)
        
    def _draw_curved_path(self, difficulty: float):
        """绘制弯道"""
        curve_center = (30, 20)
        curve_radius = 20.0
        
        # 弯道边界
        inner_curve = Circle(curve_center, curve_radius - 3, 
                           facecolor='gray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(inner_curve)
        
        outer_curve = Circle(curve_center, curve_radius + 3, 
                           facecolor='gray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(outer_curve)
        
        # 目标点
        goal_x = curve_center[0] + curve_radius * np.cos(np.pi/4)
        goal_y = curve_center[1] + curve_radius * np.sin(np.pi/4)
        self.goal = Circle((goal_x, goal_y), 3, facecolor='green', 
                          edgecolor='darkgreen', linewidth=2, alpha=0.8)
        self.ax_scenario.add_patch(self.goal)
        
    def _draw_intersection(self, difficulty: float):
        """绘制交叉路口"""
        # 主道路
        main_road = Rectangle((-4, 0), 8, 50, 
                            facecolor='lightgray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(main_road)
        
        # 交叉道路
        cross_road = Rectangle((15, 20), 20, 8, 
                             facecolor='lightgray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(cross_road)
        
        # 目标点
        self.goal = Circle((0, 45), 3, facecolor='green', 
                          edgecolor='darkgreen', linewidth=2, alpha=0.8)
        self.ax_scenario.add_patch(self.goal)
        
        # 交通车辆
        traffic_count = int(3 * difficulty)
        for i in range(traffic_count):
            x = 20 + i * 5
            y = 22
            vehicle = Rectangle((x-1.5, y-1.5), 3, 3, 
                              facecolor='yellow', edgecolor='orange', 
                              linewidth=2, alpha=0.8)
            self.ax_scenario.add_patch(vehicle)
            self.obstacles.append(vehicle)
            
    def _draw_narrow_passage(self, difficulty: float):
        """绘制窄道"""
        passage_width = max(3.0, 6.0 - 2.0 * difficulty)
        
        # 左边界
        left_boundary = Rectangle((-passage_width/2, 0), 0.5, 50, 
                                facecolor='gray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(left_boundary)
        
        # 右边界
        right_boundary = Rectangle((passage_width/2, 0), 0.5, 50, 
                                 facecolor='gray', edgecolor='black', linewidth=2)
        self.ax_scenario.add_patch(right_boundary)
        
        # 目标点
        self.goal = Circle((0, 45), 2, facecolor='green', 
                          edgecolor='darkgreen', linewidth=2, alpha=0.8)
        self.ax_scenario.add_patch(self.goal)
        
    def update_ego_vehicle(self, x: float, y: float, yaw: float, v: float = 0.0):
        """更新自车位置"""
        self.ego_position = (x, y)
        self.ego_yaw = yaw
        
        # 移除之前的自车
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.remove()
            except:
                pass
        
        # 创建新的自车
        vehicle_length = 3.0
        vehicle_width = 1.5
        
        # 计算车辆四个角点
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
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
        self.ax_scenario.add_patch(self.ego_vehicle)
        
        # 记录轨迹
        self.trajectory.append((x, y))
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            trajectory_x = [point[0] for point in self.trajectory]
            trajectory_y = [point[1] for point in self.trajectory]
            self.ax_scenario.plot(trajectory_x, trajectory_y, 'b-', 
                                alpha=0.6, linewidth=1)
        
        # 添加速度箭头
        if v > 0.1:
            arrow_length = v * 3
            arrow_end_x = x + arrow_length * cos_yaw
            arrow_end_y = y + arrow_length * sin_yaw
            
            self.ax_scenario.annotate('', xy=(arrow_end_x, arrow_end_y), 
                                    xytext=(x, y),
                                    arrowprops=dict(arrowstyle='->', 
                                                  color='red', lw=2))
        
    def update_performance(self, episode_reward: float, success: bool, 
                          collision: bool, steps: int):
        """更新性能指标"""
        self.episode_rewards.append(episode_reward)
        
        # 更新成功率
        if success:
            self.success_rates.append(1.0)
        else:
            self.success_rates.append(0.0)
            
        # 更新碰撞率
        if collision:
            self.collision_rates.append(1.0)
        else:
            self.collision_rates.append(0.0)
        
        # 更新图表
        self._update_performance_plots()
        
    def _update_performance_plots(self):
        """更新性能图表"""
        # 奖励曲线
        self.ax_rewards.clear()
        self.ax_rewards.set_title('奖励曲线', fontsize=12)
        self.ax_rewards.set_xlabel('回合数')
        self.ax_rewards.set_ylabel('奖励')
        self.ax_rewards.grid(True, alpha=0.3)
        
        if self.episode_rewards:
            episodes = range(len(self.episode_rewards))
            self.ax_rewards.plot(episodes, self.episode_rewards, 'b-', linewidth=2)
            
            # 移动平均
            if len(self.episode_rewards) > 10:
                window = min(20, len(self.episode_rewards))
                moving_avg = np.convolve(self.episode_rewards, 
                                       np.ones(window)/window, mode='valid')
                self.ax_rewards.plot(range(window-1, len(self.episode_rewards)), 
                                   moving_avg, 'r-', linewidth=2, alpha=0.7)
        
        # 性能指标
        self.ax_performance.clear()
        self.ax_performance.set_title('性能指标', fontsize=12)
        self.ax_performance.set_xlabel('回合数')
        self.ax_performance.set_ylabel('成功率/碰撞率')
        self.ax_performance.grid(True, alpha=0.3)
        
        if self.success_rates and self.collision_rates:
            episodes = range(len(self.success_rates))
            
            # 计算移动平均
            window = min(10, len(self.success_rates))
            if len(self.success_rates) > window:
                success_avg = np.convolve(self.success_rates, 
                                        np.ones(window)/window, mode='valid')
                collision_avg = np.convolve(self.collision_rates, 
                                          np.ones(window)/window, mode='valid')
                
                self.ax_performance.plot(range(window-1, len(self.success_rates)), 
                                       success_avg, 'g-', linewidth=2, label='成功率')
                self.ax_performance.plot(range(window-1, len(self.collision_rates)), 
                                       collision_avg, 'r-', linewidth=2, label='碰撞率')
                self.ax_performance.legend()
        
    def update_actions(self, action: np.ndarray):
        """更新动作分布"""
        self.action_history.append(action)
        
        self.ax_actions.clear()
        self.ax_actions.set_title('动作分布', fontsize=12)
        self.ax_actions.set_xlabel('左履带速度')
        self.ax_actions.set_ylabel('右履带速度')
        self.ax_actions.grid(True, alpha=0.3)
        
        if self.action_history:
            actions = np.array(self.action_history)
            self.ax_actions.scatter(actions[:, 0], actions[:, 1], 
                                  alpha=0.6, s=20, c='blue')
            
            # 添加当前动作
            if len(actions) > 0:
                self.ax_actions.scatter(actions[-1, 0], actions[-1, 1], 
                                      s=100, c='red', marker='x', linewidth=3)
        
        # 设置坐标轴范围
        self.ax_actions.set_xlim(-1.2, 1.2)
        self.ax_actions.set_ylim(-1.2, 1.2)
        
    def update_mcts_stats(self, simulations: int, depth: int, value: float):
        """更新MCTS统计"""
        self.mcts_stats.append({
            'simulations': simulations,
            'depth': depth,
            'value': value
        })
        
        self.ax_mcts.clear()
        self.ax_mcts.set_title('MCTS统计', fontsize=12)
        self.ax_mcts.set_xlabel('时间步')
        self.ax_mcts.set_ylabel('模拟次数')
        self.ax_mcts.grid(True, alpha=0.3)
        
        if self.mcts_stats:
            steps = range(len(self.mcts_stats))
            simulations = [stat['simulations'] for stat in self.mcts_stats]
            depths = [stat['depth'] for stat in self.mcts_stats]
            
            self.ax_mcts.plot(steps, simulations, 'b-', linewidth=2, label='模拟次数')
            self.ax_mcts.plot(steps, depths, 'g-', linewidth=2, label='搜索深度')
            self.ax_mcts.legend()
        
    def update_statistics(self, episode: int, total_episodes: int, 
                         avg_reward: float, success_rate: float, 
                         collision_rate: float):
        """更新统计信息"""
        self.ax_statistics.clear()
        self.ax_statistics.set_title('训练统计', fontsize=12)
        self.ax_statistics.axis('off')
        
        stats_text = f"""
当前回合: {episode}/{total_episodes}
场景类型: {self.current_scenario}
平均奖励: {avg_reward:.2f}
成功率: {success_rate:.2%}
碰撞率: {collision_rate:.2%}
自车位置: ({self.ego_position[0]:.1f}, {self.ego_position[1]:.1f})
自车航向: {self.ego_yaw:.2f} rad
轨迹长度: {len(self.trajectory)}
        """.strip()
        
        self.ax_statistics.text(0.1, 0.9, stats_text, 
                              transform=self.ax_statistics.transAxes,
                              verticalalignment='top',
                              fontsize=10,
                              bbox=dict(boxstyle='round', 
                                      facecolor='lightblue', 
                                      alpha=0.8))
        
    def start_animation(self):
        """启动动画"""
        self.animation = animation.FuncAnimation(
            self.fig, self._animate, interval=100, blit=False, repeat=True
        )
        os.makedirs('outputs', exist_ok=True)
        out_path = os.path.join('outputs', 'realtime_demo.gif')
        try:
            self.animation.save(out_path, writer='pillow', fps=10)
            print(f"已保存实时演示为: {out_path}")
        except Exception as e:
            print(f"保存GIF失败: {e}")
        try:
            plt.show()
        except Exception:
            pass
        
    def _animate(self, frame):
        """动画更新函数"""
        # 这里可以添加实时数据更新逻辑
        pass
        
    def simulate_training(self, num_episodes: int = 100):
        """模拟训练过程"""
        scenarios = ['straight_path', 'curved_path', 'intersection', 'narrow_passage']
        
        for episode in range(num_episodes):
            # 随机选择场景
            scenario = np.random.choice(scenarios)
            difficulty = np.random.uniform(0.5, 2.0)
            
            # 更新场景
            self.update_scenario(scenario, difficulty)
            
            # 模拟回合
            episode_reward = 0
            success = False
            collision = False
            
            for step in range(100):
                # 模拟智能体运动
                x = step * 0.5 + np.random.normal(0, 0.1)
                y = step * 0.3 + np.random.normal(0, 0.1)
                yaw = step * 0.01 + np.random.normal(0, 0.05)
                v = 2.0 + np.random.normal(0, 0.2)
                
                # 更新自车位置
                self.update_ego_vehicle(x, y, yaw, v)
                
                # 模拟动作
                action = np.random.uniform(-1, 1, 2)
                self.update_actions(action)
                
                # 模拟MCTS统计
                simulations = np.random.randint(50, 150)
                depth = np.random.randint(5, 20)
                value = np.random.normal(0, 10)
                self.update_mcts_stats(simulations, depth, value)
                
                # 计算奖励
                reward = 10 - abs(x) * 0.1 - abs(y - step * 0.3) * 0.1
                episode_reward += reward
                
                # 检查终止条件
                if y > 45:
                    success = True
                    break
                elif abs(x) > 4:
                    collision = True
                    break
                
                # 更新性能
                self.update_performance(episode_reward, success, collision, step)
                
                # 更新统计信息
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                success_rate = np.mean(self.success_rates) if self.success_rates else 0
                collision_rate = np.mean(self.collision_rates) if self.collision_rates else 0
                
                self.update_statistics(episode, num_episodes, avg_reward, 
                                     success_rate, collision_rate)
                
                # 暂停一下
                plt.pause(0.01)
            
            # 回合结束
            if not success and not collision:
                collision = True
                
            self.update_performance(episode_reward, success, collision, 100)
            
            print(f"回合 {episode}: 奖励={episode_reward:.2f}, "
                  f"成功={success}, 碰撞={collision}")


def main():
    """主函数"""
    print("=== 实时训练可视化演示 ===")
    
    # 创建可视化器
    visualizer = RealtimeTrainingVisualizer()
    
    # 启动动画
    visualizer.start_animation()
    
    # 模拟训练
    visualizer.simulate_training(50)


if __name__ == '__main__':
    main()
