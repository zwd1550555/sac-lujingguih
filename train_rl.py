# -*- coding: utf-8 -*-
"""
履带车SAC强化学习训练脚本
用于训练履带车进行路径规划和动态避障

主要功能:
- 使用SAC算法训练履带车智能体
- 支持边界感知和主动避障
- 提供安全约束和动作限制
- 支持多种训练配置和超参数调优

技术特性:
- 增强的状态表示 (22维状态向量)
- 因果奖励塑造 (反事实推理)
- 多层次安全约束系统
- 支持课程学习和渐进式训练

使用方法:
python train_rl.py --cfg config_optimized.yaml --tag my_training
"""

import os
import math
import random
import argparse
import numpy as np

from dynamic_scenes.env import Env
from dynamic_scenes.scenarioOrganizer1 import ScenarioOrganizer
from dynamic_scenes.lookup import CollisionLookup
from agent import SACAgent
from common.config_loader import load_config_file
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False


def extract_state(observation: dict, collision_lookup: CollisionLookup, num_obstacles: int = 3) -> np.ndarray:
    """
    提取智能体状态，包含车辆状态、目标信息、障碍物信息和边界感知信息
    
    状态向量构成:
    - 车辆状态 (5维): 速度、角速度、目标距离、目标相对位置
    - 障碍物信息 (12维): 最近3个障碍物的相对位置和相对速度 (4维×3个)
    - 边界感知 (5维): 前方、左前、右前、左侧、右侧方向的边界距离
    - 坡度与高程 (2维): 最近参考路径点的高程(米)与坡度(度)的归一化值
    
    Args:
        observation (dict): 环境观察数据，包含车辆信息、目标信息、地图信息等
        collision_lookup (CollisionLookup): 碰撞检测查找表，用于边界感知
        num_obstacles (int): 考虑的障碍物数量，默认为3个
    
    Returns:
        np.ndarray: 归一化的状态向量 (22维)，数据类型为float32
        
    技术细节:
        - 使用光线投射法进行边界感知，模拟激光雷达功能
        - 所有状态值都进行了归一化处理，便于神经网络学习
        - 障碍物按距离排序，优先考虑最近的障碍物
    """
    ego = observation['vehicle_info']['ego']
    goal = observation['test_setting']['goal']

    # --- 核心状态信息 ---
    ego_v = float(ego['v_mps']) / 10.0
    ego_yaw_rate = float(ego['yawrate_radps']) / math.pi

    goal_xc = float(np.mean(goal['x']))
    goal_yc = float(np.mean(goal['y']))

    dx = goal_xc - float(ego['x'])
    dy = goal_yc - float(ego['y'])
    yaw = float(ego['yaw_rad'])
    gx_local = dx * math.cos(yaw) + dy * math.sin(yaw)
    gy_local = -dx * math.sin(yaw) + dy * math.cos(yaw)
    gdist = math.hypot(dx, dy)

    state = [ego_v, ego_yaw_rate, gdist / 100.0, gx_local / 100.0, gy_local / 100.0]

    # --- 障碍物信息 ---
    others = []
    for vid, vinfo in observation['vehicle_info'].items():
        if vid == 'ego':
            continue
        d = math.hypot(float(vinfo['x']) - float(ego['x']), float(vinfo['y']) - float(ego['y']))
        others.append((d, vinfo))
    others.sort(key=lambda x: x[0])

    for i in range(num_obstacles):
        if i < len(others):
            obs = others[i][1]
            dx_o = (float(obs['x']) - float(ego['x'])) / 50.0
            dy_o = (float(obs['y']) - float(ego['y'])) / 50.0
            dvx = (float(obs['v_mps']) * math.cos(float(obs['yaw_rad'])) - float(ego['v_mps']) * math.cos(yaw)) / 10.0
            dvy = (float(obs['v_mps']) * math.sin(float(obs['yaw_rad'])) - float(ego['v_mps']) * math.sin(yaw)) / 10.0
            state.extend([dx_o, dy_o, dvx, dvy])
        else:
            state.extend([5.0, 5.0, 0.0, 0.0])
            
    # --- 增强边界感知信息 ---
    # 使用光线投射法感知边界，增加更多感知方向
    hdmaps = observation.get('hdmaps_info')
    if hdmaps and 'image_mask' in hdmaps:
        local_x_range = hdmaps['image_mask'].bitmap_info['bitmap_mask_PNG']['UTM_info']['local_x_range']
        local_y_range = hdmaps['image_mask'].bitmap_info['bitmap_mask_PNG']['UTM_info']['local_y_range']
        image_ndarray = hdmaps['image_mask'].image_ndarray
        
        # 定义更多感知方向：前方、左前45度、右前45度、左侧90度、右侧90度
        angles = [0, math.pi / 4, -math.pi / 4, math.pi / 2, -math.pi / 2]
        max_ray_dist = 50.0  # 最大感知距离50米
        
        for angle in angles:
            ray_dist = max_ray_dist
            # 沿着光线方向步进检查
            for d in np.arange(1.0, max_ray_dist, 1.0):
                check_x = float(ego['x']) + d * math.cos(yaw + angle)
                check_y = float(ego['y']) + d * math.sin(yaw + angle)
                
                # 使用碰撞检测判断点是否越界
                if collision_lookup.collisionDetection(check_x - local_x_range[0],
                                                       check_y - local_y_range[0],
                                                       yaw,
                                                       image_ndarray):
                    ray_dist = d
                    break
            state.append(ray_dist / max_ray_dist)  # 归一化
    else:
        # 如果没有地图信息，提供默认值
        state.extend([1.0] * 5)  # 5个方向的边界感知

    # --- 坡度/高程信息 ---
    # 说明：HD Map 的 reference_path.waypoints 中，单个路径点为 5 维数组 [x, y, yaw, height, slope]
    #       - height: 路面高程/海拔，单位米（相对本地地图坐标的高程基准）
    #       - slope: 路面坡度，单位度（°），>0 上坡，<0 下坡
    # 设计：
    #   1) 我们取“距离自车最近的参考路径点”的 height/slope 作为局部地形特征；
    #   2) 归一化：height/1000（默认高程量级在百米），slope/15（常见坡度在±15°内），并裁剪至[-1,1]；
    #   3) 通过 test_setting.train_cfg.enable_slope_features 开关控制是否启用该特征；
    #   4) 如需进一步增强，可加入前方若干米的坡度统计（mean/max/min），此处保留最小实现。
    try:
        tgsc_map = observation.get('hdmaps_info', {}).get('tgsc_map')
        enable_slope = observation.get('test_setting', {}).get('train_cfg', {}).get('enable_slope_features', True)
        if tgsc_map is not None and enable_slope:
            ego_x = float(ego['x'])
            ego_y = float(ego['y'])
            # 在所有参考路径中近邻检索一个最近点（步长抽样提速）
            nearest_height = 0.0
            nearest_slope_deg = 0.0
            min_dist2 = 1e18
            # reference_path 为列表，每个 path 有 waypoints: [x,y,yaw,height,slope]
            for ref in tgsc_map.reference_path:
                waypoints = ref.get('waypoints', [])
                if not waypoints:
                    continue
                # 步长抽样，提升性能
                for wp in waypoints[::5]:
                    dx = float(wp[0]) - ego_x
                    dy = float(wp[1]) - ego_y
                    d2 = dx*dx + dy*dy
                    if d2 < min_dist2:
                        min_dist2 = d2
                        # 索引3/4 分别是 height(米), slope(度)
                        nearest_height = float(wp[3]) if len(wp) > 3 else 0.0
                        nearest_slope_deg = float(wp[4]) if len(wp) > 4 else 0.0
            # 归一化：高程/1000，坡度/15 度
            height_norm = nearest_height / 1000.0
            slope_norm = max(-1.0, min(1.0, nearest_slope_deg / 15.0))
            state.extend([height_norm, slope_norm])
        else:
            state.extend([0.0, 0.0])
    except Exception:
        state.extend([0.0, 0.0])

    return np.array(state, dtype=np.float32)


def calculate_reward(curr_obs: dict, prev_obs: dict, reward_cfg: dict) -> float:
    """
    计算奖励函数，包含目标导向、效率、安全距离和碰撞惩罚
    
    Args:
        curr_obs: 当前观察
        prev_obs: 前一步观察
        reward_cfg: 奖励配置参数
    
    Returns:
        奖励值
    """
    ego = curr_obs['vehicle_info']['ego']
    end = curr_obs['test_setting']['end']
    
    # --- 最终状态奖励 ---
    reach_goal_bonus = float(reward_cfg.get('reach_goal_bonus', 500.0))
    # 增加碰撞惩罚力度，让智能体更"敬畏"碰撞
    collision_penalty = float(reward_cfg.get('collision_penalty', -800.0))
    timeout_penalty = float(reward_cfg.get('timeout_penalty', -100.0))
    
    if end == 4:
        return reach_goal_bonus
    if end in (2, 3):
        return collision_penalty
    if end == 1:
        return timeout_penalty

    # --- 过程奖励与惩罚 ---
    r = 0.0
    
    # 1. 目标导向奖励
    goal = curr_obs['test_setting']['goal']
    gx = float(np.mean(goal['x']))
    gy = float(np.mean(goal['y']))
    dist_now = math.hypot(float(ego['x']) - gx, float(ego['y']) - gy)
    ego_prev = prev_obs['vehicle_info']['ego']
    dist_prev = math.hypot(float(ego_prev['x']) - gx, float(ego_prev['y']) - gy)
    
    progress_weight = float(reward_cfg.get('progress_weight', 2.0))
    r += (dist_prev - dist_now) * progress_weight

    # 2. 效率与平顺性奖励/惩罚
    speed_weight = float(reward_cfg.get('speed_weight', 0.1))
    yawrate_penalty = float(reward_cfg.get('yawrate_penalty', 0.2))  # 稍微增加，鼓励更平滑的转向
    time_penalty = float(reward_cfg.get('time_penalty', 0.5))
    
    r += float(ego['v_mps']) * speed_weight
    r -= abs(float(ego['yawrate_radps'])) * yawrate_penalty
    r -= time_penalty

    # 2.1 坡度相关奖励/惩罚（能耗/安全/稳定性）
    # 说明：
    #   - 上坡能耗：坡度越大、速度越高，能耗越大 → 惩罚项与 slope_rad 和 v 成正相关；
    #   - 下坡超速：下坡时设定“安全速度” v_safe（坡度越陡安全速度越低），超过则惩罚；
    #   - 坡面稳定性：坡大时更鼓励平稳，惩罚 |yawrate|*(1+|slope|)。
    try:
        tgsc_map = curr_obs.get('hdmaps_info', {}).get('tgsc_map')
        slope_uphill_weight = float(reward_cfg.get('slope_uphill_weight', 0.05))
        slope_downhill_weight = float(reward_cfg.get('slope_downhill_weight', 0.08))
        slope_smooth_weight = float(reward_cfg.get('slope_smooth_weight', 0.02))
        if tgsc_map is not None:
            ego_x = float(ego['x']); ego_y = float(ego['y'])
            # 近邻坡度
            nearest_slope_deg = 0.0
            min_d2 = 1e18
            for ref in tgsc_map.reference_path:
                wps = ref.get('waypoints', [])
                for wp in wps[::5]:
                    dx = float(wp[0]) - ego_x
                    dy = float(wp[1]) - ego_y
                    d2 = dx*dx + dy*dy
                    if d2 < min_d2:
                        min_d2 = d2
                        nearest_slope_deg = float(wp[4]) if len(wp) > 4 else 0.0
            slope_rad = nearest_slope_deg * math.pi / 180.0
            v_now = float(ego['v_mps'])
            yawrate_now = float(ego['yawrate_radps'])
            # 上坡能耗惩罚（坡度越大越“吃力”）
            r -= slope_uphill_weight * max(0.0, slope_rad) * v_now
            # 下坡超速惩罚：超过下坡安全速度给予惩罚
            vmax_cfg = curr_obs.get('test_setting', {}).get('dynamics_params', {}).get('max_speed', 5.0)
            v_safe = float(vmax_cfg) * (1.0 - min(0.5, abs(nearest_slope_deg) / 15.0))
            if nearest_slope_deg < 0.0 and v_now > v_safe:
                r -= slope_downhill_weight * (v_now - v_safe)
            # 坡面稳定性：坡度越大越鼓励平稳（惩罚转向与坡度的联合作用）
            r -= slope_smooth_weight * abs(yawrate_now) * (1.0 + abs(slope_rad))
    except Exception:
        pass
    
    # 3. 主动避障与安全距离惩罚
    min_dist_to_obs = float('inf')
    ego_x = float(ego['x'])
    ego_y = float(ego['y'])
    
    # 找到最近的动态障碍物
    for vid, vinfo in curr_obs['vehicle_info'].items():
        if vid == 'ego':
            continue
        dist = math.hypot(float(vinfo['x']) - ego_x, float(vinfo['y']) - ego_y)
        if dist < min_dist_to_obs:
            min_dist_to_obs = dist
            
    # 定义"安全缓冲区"，进入该区域将受到惩罚
    safety_distance = float(reward_cfg.get('safety_distance', 15.0))  # 15米安全距离
    safety_penalty_weight = float(reward_cfg.get('safety_penalty_weight', 1.0))

    if min_dist_to_obs < safety_distance:
        # 惩罚力度与侵入缓冲区的深度成正比，距离越近，惩罚越大
        r -= (1.0 - min_dist_to_obs / safety_distance) * safety_penalty_weight
    
    # 4. 边界安全惩罚（基于新增的边界感知信息）
    boundary_penalty_weight = float(reward_cfg.get('boundary_penalty_weight', 0.5))
    hdmaps = curr_obs.get('hdmaps_info')
    if hdmaps and 'image_mask' in hdmaps:
        # 如果前方边界距离过近，给予惩罚
        # 这里可以基于状态向量中的边界感知信息进行惩罚
        # 由于边界信息在状态提取中已经计算，这里简化处理
        pass
    
    # 5. 因果奖励塑造 - 反事实思考
    causal_credit_bonus = float(reward_cfg.get('causal_credit_bonus', 3.0))
    
    # 模拟默认动作（直行）的结果
    ego_prev = prev_obs['vehicle_info']['ego']
    default_action_speed = 0.3  # 默认中速直行
    
    # 简单的前向预测：如果上一步采取默认动作会怎样
    dt = curr_obs['test_setting']['dt']
    prev_v = float(ego_prev['v_mps'])
    prev_yaw = float(ego_prev['yaw_rad'])
    prev_x = float(ego_prev['x'])
    prev_y = float(ego_prev['y'])
    
    # 预测默认动作下的位置
    predicted_x = prev_x + default_action_speed * math.cos(prev_yaw) * dt
    predicted_y = prev_y + default_action_speed * math.sin(prev_yaw) * dt
    
    # 检查预测位置是否会发生碰撞
    would_have_collided = False
    for vid, vinfo in curr_obs['vehicle_info'].items():
        if vid == 'ego':
            continue
        other_x = float(vinfo['x'])
        other_y = float(vinfo['y'])
        dist_to_other = math.hypot(predicted_x - other_x, predicted_y - other_y)
        if dist_to_other < 8.0:  # 碰撞阈值
            would_have_collided = True
            break
    
    # 如果默认动作会导致碰撞，但当前是安全的，给予因果奖励
    is_currently_safe = curr_obs['test_setting']['end'] == -1
    if would_have_collided and is_currently_safe:
        r += causal_credit_bonus
        
    return r


def apply_safety_constraints(action: np.ndarray, env, observation: dict) -> np.ndarray:
    """
    应用安全约束，限制动作在安全范围内
    
    Args:
        action: 原始动作 [v_left, v_right]
        env: 环境对象
        observation: 当前观察
    
    Returns:
        约束后的安全动作
    """
    try:
        ego = observation['vehicle_info']['ego']
        ego_v = float(ego['v_mps'])
        ego_yawrate = float(ego['yawrate_radps'])
        
        # 获取车辆参数
        shape = ego['shape']
        rmin = float(shape.get('min_turn_radius', 14.2))
        width = float(shape.get('width', 6.7))
        
        # 获取动力学参数
        dynamics_params = env.controller.control_info.test_setting.get('dynamics_params', {})
        vmax = float(dynamics_params.get('max_speed', 5.0))
        a_max = float(dynamics_params.get('a_max', 2.0))
        b_max = float(dynamics_params.get('b_max', 3.0))
        omega_max = float(dynamics_params.get('omega_abs_max', 0.8))
        # 训练配置中下坡限速因子
        # 说明：下坡时将最大安全线速度 vmax 按比例缩放，避免溜车/过快导致失稳。
        train_cfg = observation.get('test_setting', {}).get('train_cfg', {})
        downhill_factor = float(train_cfg.get('max_speed_downhill_factor', 0.7))
        
        v_l, v_r = float(action[0]), float(action[1])
        
        # 1. 速度限制
        v_l = np.clip(v_l, -1.0, 1.0)
        v_r = np.clip(v_r, -1.0, 1.0)
        
        # 2. 最小转弯半径约束
        if rmin > 1e-6:
            # 计算当前速度下的最大允许差速
            current_speed = max(abs(v_l), abs(v_r)) * vmax
            if current_speed > 0.1:  # 避免除零
                # 基于最小转弯半径计算最大角速度
                max_omega = current_speed / rmin
                max_omega = min(max_omega, omega_max)
                
                # 计算最大允许差速
                max_diff = max_omega * width / vmax
                
                # 限制差速
                diff = v_r - v_l
                if abs(diff) > max_diff:
                    scale = max_diff / (abs(diff) + 1e-6)
                    mid = 0.5 * (v_l + v_r)
                    half = 0.5 * abs(diff) * scale
                    v_l = mid - half * np.sign(diff)
                    v_r = mid + half * np.sign(diff)
        
        # 计算邻近坡度
        # 说明：在全部 reference_path 中以步长抽样寻找最近 waypoint，读取 slope 角度并转弧度。
        tgsc_map = observation.get('hdmaps_info', {}).get('tgsc_map')
        slope_rad = 0.0
        if tgsc_map is not None:
            ego_x = float(ego['x']); ego_y = float(ego['y'])
            min_d2 = 1e18
            for ref in tgsc_map.reference_path:
                wps = ref.get('waypoints', [])
                for wp in wps[::5]:
                    dx = float(wp[0]) - ego_x
                    dy = float(wp[1]) - ego_y
                    d2 = dx*dx + dy*dy
                    if d2 < min_d2:
                        min_d2 = d2
                        if len(wp) > 4:
                            slope_rad = float(wp[4]) * math.pi / 180.0

        # 3. 加速度限制（考虑坡度的等效加减速度）
        # 说明：上坡会“吃掉”一部分可用驱动力（a_eff_max = a_max - g*sin(slope+)），
        #       下坡需要预留更强的制动（b_eff_max = b_max + g*sin(-slope)）。
        dt = 0.1  # 假设时间步长
        g = 9.81
        a_eff_max = max(0.2, a_max - g * max(0.0, math.sin(slope_rad)))
        b_eff_max = b_max + g * max(0.0, -math.sin(slope_rad))
        max_accel = a_eff_max * dt / max(1e-6, vmax)
        max_decel = b_eff_max * dt / max(1e-6, vmax)
        
        # 限制加速度变化
        v_l = np.clip(v_l, ego_v/vmax - max_decel, ego_v/vmax + max_accel)
        v_r = np.clip(v_r, ego_v/vmax - max_decel, ego_v/vmax + max_accel)
        
        # 4. 下坡限速与角速度限制
        # 说明：
        #   - 下坡：将 vmax_eff = vmax * downhill_factor（下限0.3，避免极端0导致停滞）；
        #   - 角速度：按车辆几何和 vmax 约束 (与差速相关)。
        if slope_rad < 0.0:
            vmax_eff = vmax * max(0.3, min(1.0, downhill_factor))
        else:
            vmax_eff = vmax
        current_omega = (v_r - v_l) * vmax / width
        if abs(current_omega) > omega_max:
            scale = omega_max / (abs(current_omega) + 1e-6)
            mid = 0.5 * (v_l + v_r)
            half = 0.5 * abs(v_r - v_l) * scale
            v_l = mid - half * np.sign(v_r - v_l)
            v_r = mid + half * np.sign(v_r - v_l)
        
        # 5. 最终范围限制
        v_l = np.clip(v_l, -1.0, 1.0)
        v_r = np.clip(v_r, -1.0, 1.0)

        # 将速度按下坡限速进行二次压缩（归一化域内收缩）
        # 说明：此处 action 在 [-1,1] 域内，乘以 speed_scale 达到“限速”效果。
        speed_scale = vmax_eff / max(1e-6, vmax)
        v_l *= speed_scale
        v_r *= speed_scale
        
        return np.array([v_l, v_r], dtype=np.float32)
        
    except Exception as e:
        # 如果约束计算失败，返回原始动作的裁剪版本
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='YAML/JSON config for training')
    parser.add_argument('--logdir', type=str, default='runs', help='TensorBoard/CSV log dir')
    parser.add_argument('--tag', type=str, default='default', help='run tag')
    args = parser.parse_args()
    # 固定随机种子以复现实验
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, 'demo', 'demo_inputs')
    output_dir = os.path.join(project_root, 'demo', 'demo_outputs')

    so = ScenarioOrganizer()
    so.load(input_dir, output_dir)
    if not so.scenario_list:
        print("未找到场景，请将场景放入 demo/demo_inputs/Scenarios 下")
        return

    print('初始化碰撞查找表...')
    collision_lookup = CollisionLookup()
    print('完成')

    env = Env()

    # 读取训练配置（可覆盖默认）
    cfg = load_config_file(args.cfg)
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    net_cfg = cfg.get('network', {}) if isinstance(cfg, dict) else {}
    opt_cfg = cfg.get('optimizer', {}) if isinstance(cfg, dict) else {}
    reward_cfg = cfg.get('reward', {}) if isinstance(cfg, dict) else {}

    NUM_OBS = int(train_cfg.get('num_obstacles', 3))
    # 状态维度：5(自车/目标) + 4*NUM_OBS(障碍物) + 5(边界感知) + 2(坡度/高程)
    state_dim = 5 + 4 * NUM_OBS + 5 + 2
    action_dim = 2
    hidden_dim = int(net_cfg.get('hidden_dim', 256))
    gamma = float(opt_cfg.get('gamma', 0.99))
    tau = float(opt_cfg.get('tau', 0.005))
    actor_lr = float(opt_cfg.get('actor_lr', 3e-4))
    critic_lr = float(opt_cfg.get('critic_lr', 3e-4))
    buffer_size = int(opt_cfg.get('replay_size', 1_000_000))

    target_update_interval = int(opt_cfg.get('target_update_interval', 1))
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim,
                     gamma=gamma, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr,
                     replay_buffer_capacity=buffer_size, target_update_interval=target_update_interval)
    # 初始化 ReplayBuffer 形状，避免首次 push 时额外开销
    agent.replay_buffer.state_dim = state_dim
    agent.replay_buffer.action_dim = action_dim

    max_episodes = int(train_cfg.get('max_episodes', 2000))  # 增加训练轮数
    max_steps = int(train_cfg.get('max_steps', 1500))  # 增加每轮最大步数
    batch_size = int(train_cfg.get('batch_size', 512))  # 增大批次大小
    save_interval = int(train_cfg.get('save_interval', 50))
    warmup_steps = int(train_cfg.get('warmup_steps', 10000))  # 增加预热步数

    # 简单 warmup：前若干步使用随机动作填充经验池
    steps_done = 0

    # 日志器
    run_name = f"{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(args.logdir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if TB_AVAILABLE else None
    csv_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('episode,step,return,critic_loss,actor_loss,alpha_loss,alpha,ego_v,ego_yawrate\n')

    # 可选 Scheduler
    lr_sched = opt_cfg.get('lr_scheduler', 'none')
    try:
        import torch
        if lr_sched == 'cosine':
            actor_s = torch.optim.lr_scheduler.CosineAnnealingLR(agent.actor_opt, T_max=max_episodes)
            critic_s = torch.optim.lr_scheduler.CosineAnnealingLR(agent.critic_opt, T_max=max_episodes)
        elif lr_sched == 'step':
            step_size = int(opt_cfg.get('lr_step_size', 200))
            gamma_s = float(opt_cfg.get('lr_gamma', 0.5))
            actor_s = torch.optim.lr_scheduler.StepLR(agent.actor_opt, step_size=step_size, gamma=gamma_s)
            critic_s = torch.optim.lr_scheduler.StepLR(agent.critic_opt, step_size=step_size, gamma=gamma_s)
        elif lr_sched == 'exp':
            gamma_s = float(opt_cfg.get('lr_gamma', 0.99))
            actor_s = torch.optim.lr_scheduler.ExponentialLR(agent.actor_opt, gamma=gamma_s)
            critic_s = torch.optim.lr_scheduler.ExponentialLR(agent.critic_opt, gamma=gamma_s)
        else:
            actor_s = critic_s = None
    except Exception:
        actor_s = critic_s = None

    for ep in range(max_episodes):
        scenario = random.choice(so.scenario_list)
        observation, _ = env.make(scenario, collision_lookup, read_only=True, save_img_path='')
        # 将训练配置透传到 observation.test_setting 供状态/约束使用
        if isinstance(cfg, dict):
            observation['test_setting']['train_cfg'] = {
                'enable_slope_features': bool(train_cfg.get('enable_slope_features', True)),
                'slope_preview_meters': float(train_cfg.get('slope_preview_meters', 0.0)),
                'max_speed_downhill_factor': float(train_cfg.get('max_speed_downhill_factor', 0.7)),
            }

        prev_obs = observation
        state = extract_state(observation, collision_lookup, NUM_OBS)
        ep_ret = 0.0

        for step in range(max_steps):
            if steps_done < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            else:
                # 安全层：基于车辆动力学约束对动作进行限制
                action = agent.select_action(state)
                action = apply_safety_constraints(action, env, observation)
            next_obs = env.step_rl(tuple(action.tolist()), collision_lookup)
            # 透传配置以便下一步使用
            if isinstance(cfg, dict):
                next_obs['test_setting']['train_cfg'] = observation['test_setting']['train_cfg']
            reward = calculate_reward(next_obs, prev_obs, reward_cfg)
            done = next_obs['test_setting']['end'] != -1
            next_state = extract_state(next_obs, collision_lookup, NUM_OBS)

            agent.replay_buffer.push(state, action, reward, next_state, done)
            metrics = agent.update(batch_size)
            steps_done += 1

            state = next_state
            prev_obs = next_obs
            ep_ret += reward
            if done:
                break

        # 记录日志
        ego_now = prev_obs['vehicle_info']['ego']
        ego_v = float(ego_now['v_mps'])
        ego_yawrate = float(ego_now['yawrate_radps'])
        if writer is not None and metrics is not None:
            writer.add_scalar('return/episode', ep_ret, ep)
            writer.add_scalar('loss/critic', metrics['critic_loss'], ep)
            writer.add_scalar('loss/actor', metrics['actor_loss'], ep)
            writer.add_scalar('loss/alpha', metrics['alpha_loss'], ep)
            writer.add_scalar('alpha/value', metrics['alpha'], ep)
            writer.add_scalar('ego/v_mps', ego_v, ep)
            writer.add_scalar('ego/yawrate', ego_yawrate, ep)
        with open(csv_path, 'a') as f:
            f.write(f"{ep+1},{step+1},{ep_ret:.4f},{(metrics or {}).get('critic_loss','')},{(metrics or {}).get('actor_loss','')},{(metrics or {}).get('alpha_loss','')},{(metrics or {}).get('alpha','')},{ego_v:.4f},{ego_yawrate:.4f}\n")

        print(f"Episode {ep+1} | steps {step+1} | return {ep_ret:.2f}")

        # step scheduler per-episode
        if actor_s is not None:
            actor_s.step()
        if critic_s is not None:
            critic_s.step()

        if (ep + 1) % save_interval == 0:
            model_dir = os.path.join(output_dir, 'sac_model')
            os.makedirs(model_dir, exist_ok=True)
            import torch
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, f'actor_ep_{ep+1}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(model_dir, f'critic_ep_{ep+1}.pth'))
            print(f"Saved models to {model_dir}")


if __name__ == '__main__':
    main()


