# -*- coding: utf-8 -*-
"""
训练场景可视化演示脚本
快速展示不同类型的训练场景

使用方法:
python visualize_scenarios.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, Polygon
import matplotlib.animation as animation
from scenario_visualizer import ScenarioVisualizer


def _setup_chinese_font():
    """自动适配中文字体，避免中文显示为方块。"""
    # 按优先级排列的常见中文字体名称（不同系统/发行版）
    candidate_fonts = [
        'Noto Sans CJK SC', 'Noto Serif CJK SC',
        'Source Han Sans SC', 'Source Han Serif SC', 'Source Han Sans CN',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
        'Microsoft YaHei', 'SimHei', 'SimSun', 'DengXian',
        'PingFang SC', 'Heiti SC', 'Sarasa Gothic SC', 'LXGW WenKai'
    ]

    # 获取系统可用字体名集合
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
        # 回退到 DejaVu Sans（不含中文，便于提示安装字体）
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print('警告: 未检测到中文字体，建议安装 Noto Sans CJK 或 WenQuanYi 字体以获得正确中文显示。')

    # 解决坐标轴负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 打印当前使用的首选字体，便于排查
    try:
        print(f"中文字体设置: {plt.rcParams['font.sans-serif'][0]}")
    except Exception:
        pass


def create_scenario_demo():
    """创建场景演示"""
    # 设置中文字体（自动适配）
    _setup_chinese_font()
    
    # 创建子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    fig.suptitle('履带车智能体训练场景展示', fontsize=16, fontweight='bold')
    
    scenarios = [
        ('straight_path', '直线路径', 0, 0),
        ('curved_path', '弯道场景', 0, 1),
        ('static_obstacles', '静态障碍物', 0, 2),
        ('dynamic_obstacles', '动态障碍物', 0, 3),
        ('intersection', '交叉路口', 1, 0),
        ('narrow_passage', '窄道通行', 1, 1),
        ('complex_scenario', '复杂场景', 1, 2),
        ('mcts_planning', 'MCTS规划', 1, 3)
    ]
    
    for scenario_type, title, row, col in scenarios:
        ax = axes[row, col]
        ax.set_xlim(-30, 80)
        ax.set_ylim(-20, 60)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        if scenario_type == 'straight_path':
            _draw_straight_path(ax)
        elif scenario_type == 'curved_path':
            _draw_curved_path(ax)
        elif scenario_type == 'static_obstacles':
            _draw_static_obstacles(ax)
        elif scenario_type == 'dynamic_obstacles':
            _draw_dynamic_obstacles(ax)
        elif scenario_type == 'intersection':
            _draw_intersection(ax)
        elif scenario_type == 'narrow_passage':
            _draw_narrow_passage(ax)
        elif scenario_type == 'complex_scenario':
            _draw_complex_scenario(ax)
        elif scenario_type == 'mcts_planning':
            _draw_mcts_planning(ax)
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    out_path = os.path.join('outputs', 'scenarios_grid.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"已保存场景可视化到: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def _draw_straight_path(ax):
    """绘制直线路径场景"""
    # 道路边界
    road_width = 8.0
    road_length = 60.0
    
    # 左边界
    left_boundary = Rectangle((-road_width/2, 0), 0.5, road_length, 
                            facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(left_boundary)
    
    # 右边界
    right_boundary = Rectangle((road_width/2, 0), 0.5, road_length, 
                             facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(right_boundary)
    
    # 起始点
    start_point = Circle((0, 5), 2, facecolor='blue', edgecolor='navy', 
                       linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_point = Circle((0, 50), 3, facecolor='green', edgecolor='darkgreen', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle = Rectangle((-1.5, 15), 3, 2, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    # 轨迹
    ax.plot([0, 0], [5, 15], 'b-', linewidth=2, alpha=0.6)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 60)


def _draw_curved_path(ax):
    """绘制弯道场景"""
    # 弯道中心
    curve_center = (25, 25)
    curve_radius = 20.0
    
    # 内弯边界
    inner_curve = Circle(curve_center, curve_radius - 3, 
                       facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(inner_curve)
    
    # 外弯边界
    outer_curve = Circle(curve_center, curve_radius + 3, 
                       facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(outer_curve)
    
    # 起始点
    start_x = curve_center[0] + curve_radius * np.cos(-np.pi/3)
    start_y = curve_center[1] + curve_radius * np.sin(-np.pi/3)
    start_point = Circle((start_x, start_y), 2, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_x = curve_center[0] + curve_radius * np.cos(np.pi/3)
    goal_y = curve_center[1] + curve_radius * np.sin(np.pi/3)
    goal_point = Circle((goal_x, goal_y), 3, facecolor='green', 
                      edgecolor='darkgreen', linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle_x = curve_center[0] + curve_radius * np.cos(-np.pi/6)
    vehicle_y = curve_center[1] + curve_radius * np.sin(-np.pi/6)
    vehicle = Rectangle((vehicle_x-1.5, vehicle_y-1), 3, 2, 
                      facecolor='blue', edgecolor='navy', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)


def _draw_static_obstacles(ax):
    """绘制静态障碍物场景"""
    # 基础道路
    road_width = 8.0
    road_length = 60.0
    
    # 道路边界
    left_boundary = Rectangle((-road_width/2, 0), 0.5, road_length, 
                            facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(left_boundary)
    
    right_boundary = Rectangle((road_width/2, 0), 0.5, road_length, 
                             facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(right_boundary)
    
    # 静态障碍物
    obstacles = [(0, 20), (-2, 35), (2, 45)]
    for x, y in obstacles:
        obstacle = Rectangle((x-1.5, y-1.5), 3, 3, 
                           facecolor='red', edgecolor='darkred', 
                           linewidth=2, alpha=0.8)
        ax.add_patch(obstacle)
    
    # 起始点
    start_point = Circle((0, 5), 2, facecolor='blue', edgecolor='navy', 
                       linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_point = Circle((0, 50), 3, facecolor='green', edgecolor='darkgreen', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle = Rectangle((-1.5, 10), 3, 2, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    # 规划路径
    path_x = [0, 0, -1, -1, 1, 1, 0]
    path_y = [5, 15, 25, 30, 40, 45, 50]
    ax.plot(path_x, path_y, 'b--', linewidth=2, alpha=0.6)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 60)


def _draw_dynamic_obstacles(ax):
    """绘制动态障碍物场景"""
    # 基础道路
    road_width = 8.0
    road_length = 60.0
    
    # 道路边界
    left_boundary = Rectangle((-road_width/2, 0), 0.5, road_length, 
                            facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(left_boundary)
    
    right_boundary = Rectangle((road_width/2, 0), 0.5, road_length, 
                             facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(right_boundary)
    
    # 动态障碍物（带运动轨迹）
    obstacles = [(0, 20), (-2, 35), (2, 45)]
    for i, (x, y) in enumerate(obstacles):
        obstacle = Circle((x, y), 2, facecolor='orange', 
                        edgecolor='red', linewidth=2, alpha=0.8)
        ax.add_patch(obstacle)
        
        # 运动轨迹
        if i == 0:
            ax.plot([x, x+3], [y, y+5], 'orange', linewidth=2, alpha=0.6)
        elif i == 1:
            ax.plot([x, x-2], [y, y+3], 'orange', linewidth=2, alpha=0.6)
        else:
            ax.plot([x, x+2], [y, y+3], 'orange', linewidth=2, alpha=0.6)
    
    # 起始点
    start_point = Circle((0, 5), 2, facecolor='blue', edgecolor='navy', 
                       linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_point = Circle((0, 50), 3, facecolor='green', edgecolor='darkgreen', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle = Rectangle((-1.5, 10), 3, 2, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 60)


def _draw_intersection(ax):
    """绘制交叉路口场景"""
    # 主道路
    main_road = Rectangle((-4, 0), 8, 50, 
                        facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(main_road)
    
    # 交叉道路
    cross_road = Rectangle((10, 20), 20, 8, 
                         facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(cross_road)
    
    # 交通车辆
    traffic_vehicles = [(15, 22), (20, 24), (25, 22)]
    for x, y in traffic_vehicles:
        vehicle = Rectangle((x-1.5, y-1.5), 3, 3, 
                          facecolor='yellow', edgecolor='orange', 
                          linewidth=2, alpha=0.8)
        ax.add_patch(vehicle)
    
    # 起始点
    start_point = Circle((0, 5), 2, facecolor='blue', edgecolor='navy', 
                       linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_point = Circle((0, 45), 3, facecolor='green', edgecolor='darkgreen', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle = Rectangle((-1.5, 15), 3, 2, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    # 等待区域
    wait_area = Rectangle((8, 20), 4, 8, 
                        facecolor='yellow', edgecolor='orange', 
                        linewidth=2, alpha=0.3)
    ax.add_patch(wait_area)
    
    ax.set_xlim(-10, 40)
    ax.set_ylim(0, 50)


def _draw_narrow_passage(ax):
    """绘制窄道场景"""
    # 窄道边界
    passage_width = 4.0
    
    # 左边界
    left_boundary = Rectangle((-passage_width/2, 0), 0.5, 50, 
                            facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(left_boundary)
    
    # 右边界
    right_boundary = Rectangle((passage_width/2, 0), 0.5, 50, 
                             facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(right_boundary)
    
    # 起始点
    start_point = Circle((0, 5), 1.5, facecolor='blue', edgecolor='navy', 
                       linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_point = Circle((0, 45), 2, facecolor='green', edgecolor='darkgreen', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle = Rectangle((-1, 15), 2, 1.5, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    # 安全边界
    safety_left = Rectangle((-passage_width/2+0.5, 0), 0.2, 50, 
                          facecolor='red', edgecolor='darkred', 
                          linewidth=1, alpha=0.3)
    ax.add_patch(safety_left)
    
    safety_right = Rectangle((passage_width/2-0.7, 0), 0.2, 50, 
                           facecolor='red', edgecolor='darkred', 
                           linewidth=1, alpha=0.3)
    ax.add_patch(safety_right)
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 50)


def _draw_complex_scenario(ax):
    """绘制复杂场景"""
    # 弯道
    curve_center = (20, 20)
    curve_radius = 15.0
    
    # 弯道边界
    inner_curve = Circle(curve_center, curve_radius - 2, 
                       facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(inner_curve)
    
    outer_curve = Circle(curve_center, curve_radius + 2, 
                       facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(outer_curve)
    
    # 多个障碍物
    obstacles = [(15, 25), (25, 15), (30, 30), (10, 35)]
    for x, y in obstacles:
        obstacle = Rectangle((x-1, y-1), 2, 2, 
                          facecolor='purple', edgecolor='indigo', 
                          linewidth=2, alpha=0.8)
        ax.add_patch(obstacle)
    
    # 起始点
    start_x = curve_center[0] + curve_radius * np.cos(-np.pi/2)
    start_y = curve_center[1] + curve_radius * np.sin(-np.pi/2)
    start_point = Circle((start_x, start_y), 2, facecolor='blue', 
                       edgecolor='navy', linewidth=2, alpha=0.8)
    ax.add_patch(start_point)
    
    # 目标点
    goal_x = curve_center[0] + curve_radius * np.cos(np.pi/2)
    goal_y = curve_center[1] + curve_radius * np.sin(np.pi/2)
    goal_point = Circle((goal_x, goal_y), 3, facecolor='green', 
                      edgecolor='darkgreen', linewidth=2, alpha=0.8)
    ax.add_patch(goal_point)
    
    # 履带车
    vehicle_x = curve_center[0] + curve_radius * np.cos(-np.pi/4)
    vehicle_y = curve_center[1] + curve_radius * np.sin(-np.pi/4)
    vehicle = Rectangle((vehicle_x-1.5, vehicle_y-1), 3, 2, 
                      facecolor='blue', edgecolor='navy', 
                      linewidth=2, alpha=0.8)
    ax.add_patch(vehicle)
    
    # 复杂路径
    path_angles = np.linspace(-np.pi/2, np.pi/2, 20)
    path_x = [curve_center[0] + curve_radius * np.cos(angle) for angle in path_angles]
    path_y = [curve_center[1] + curve_radius * np.sin(angle) for angle in path_angles]
    ax.plot(path_x, path_y, 'b--', linewidth=2, alpha=0.6)
    
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)


def _draw_mcts_planning(ax):
    """绘制MCTS规划场景"""
    # 基础场景
    _draw_static_obstacles(ax)
    
    # MCTS搜索树可视化
    # 根节点
    root_x, root_y = 0, 10
    ax.plot(root_x, root_y, 'bo', markersize=8, label='根节点')
    
    # 搜索分支
    branches = [
        (root_x, root_y, 2, 15, 'b-', 0.5),
        (root_x, root_y, -2, 15, 'b-', 0.5),
        (root_x, root_y, 0, 20, 'b-', 0.5),
        (2, 15, 3, 25, 'g-', 0.7),
        (2, 15, 1, 25, 'g-', 0.7),
        (-2, 15, -3, 25, 'r-', 0.3),
        (-2, 15, -1, 25, 'r-', 0.3),
    ]
    
    for start_x, start_y, end_x, end_y, style, alpha in branches:
        ax.plot([start_x, end_x], [start_y, end_y], style, 
               linewidth=2, alpha=alpha)
        ax.plot(end_x, end_y, 'o', markersize=4, color=style[0])
    
    # 最优路径
    ax.plot([0, 1, 2, 3, 0], [10, 20, 30, 40, 50], 'g-', 
           linewidth=3, alpha=0.8, label='最优路径')
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=8)
    
    # 添加MCTS信息
    ax.text(0.02, 0.98, 'MCTS规划\n• 搜索树构建\n• 多步前瞻\n• 最优路径选择', 
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
           fontsize=8)


def create_training_progress_demo():
    """创建训练进度演示"""
    # 设置中文字体（自动适配）
    _setup_chinese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 训练曲线
    episodes = np.arange(0, 1000, 10)
    success_rate = 0.3 + 0.6 * (1 - np.exp(-episodes/200))
    collision_rate = 0.2 * np.exp(-episodes/150)
    avg_reward = -50 + 100 * (1 - np.exp(-episodes/300))
    
    ax1.plot(episodes, success_rate, 'g-', linewidth=2, label='成功率')
    ax1.plot(episodes, collision_rate, 'r-', linewidth=2, label='碰撞率')
    ax1.set_xlabel('训练回合数')
    ax1.set_ylabel('性能指标')
    ax1.set_title('训练进度曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 奖励分布
    rewards = np.random.normal(50, 20, 1000)
    rewards = np.clip(rewards, -100, 200)
    
    ax2.hist(rewards, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax2.axvline(np.mean(rewards), color='red', linestyle='--', 
               linewidth=2, label=f'平均值: {np.mean(rewards):.1f}')
    ax2.set_xlabel('奖励值')
    ax2.set_ylabel('频次')
    ax2.set_title('奖励分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    out_path = os.path.join('outputs', 'training_progress.png')
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"已保存训练进度可视化到: {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def main():
    """主函数"""
    print("=== 履带车智能体训练场景可视化 ===")
    print("1. 场景展示")
    print("2. 训练进度")
    print("3. 退出")
    
    choice = input("请选择 (1-3): ")
    
    if choice == '1':
        create_scenario_demo()
    elif choice == '2':
        create_training_progress_demo()
    elif choice == '3':
        print("退出程序")
    else:
        print("无效选择")


if __name__ == '__main__':
    main()
