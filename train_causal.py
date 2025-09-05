# -*- coding: utf-8 -*-
"""
因果强化学习增强的SAC训练脚本
集成液态神经网络、世界模型和因果推理
"""

import os
import math
import random
import argparse
import numpy as np

from dynamic_scenes.env import Env
from dynamic_scenes.scenarioOrganizer1 import ScenarioOrganizer
from dynamic_scenes.lookup import CollisionLookup
from agent_lnn import LiquidSACAgent
from world_model import WorldModel, CausalReasoner, WorldModelTrainer
from common.config_loader import load_config_file
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
    TB_AVAILABLE = True
except Exception:
    TB_AVAILABLE = False


def extract_state_causal(observation: dict, collision_lookup: CollisionLookup, num_obstacles: int = 3) -> np.ndarray:
    """
    因果增强的状态提取函数
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

    return np.array(state, dtype=np.float32)


def calculate_reward_causal_enhanced(curr_obs: dict, prev_obs: dict, reward_cfg: dict, 
                                   causal_reasoner: CausalReasoner = None) -> float:
    """
    因果增强的奖励函数
    """
    ego = curr_obs['vehicle_info']['ego']
    end = curr_obs['test_setting']['end']
    
    # --- 最终状态奖励 ---
    reach_goal_bonus = float(reward_cfg.get('reach_goal_bonus', 600.0))
    collision_penalty = float(reward_cfg.get('collision_penalty', -1000.0))
    timeout_penalty = float(reward_cfg.get('timeout_penalty', -150.0))
    
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
    
    progress_weight = float(reward_cfg.get('progress_weight', 2.5))
    r += (dist_prev - dist_now) * progress_weight

    # 2. 效率与平顺性奖励/惩罚
    speed_weight = float(reward_cfg.get('speed_weight', 0.15))
    yawrate_penalty = float(reward_cfg.get('yawrate_penalty', 0.25))
    time_penalty = float(reward_cfg.get('time_penalty', 0.6))
    
    r += float(ego['v_mps']) * speed_weight
    r -= abs(float(ego['yawrate_radps'])) * yawrate_penalty
    r -= time_penalty
    
    # 3. 主动避障与安全距离惩罚
    min_dist_to_obs = float('inf')
    ego_x = float(ego['x'])
    ego_y = float(ego['y'])
    
    for vid, vinfo in curr_obs['vehicle_info'].items():
        if vid == 'ego':
            continue
        dist = math.hypot(float(vinfo['x']) - ego_x, float(vinfo['y']) - ego_y)
        if dist < min_dist_to_obs:
            min_dist_to_obs = dist
            
    safety_distance = float(reward_cfg.get('safety_distance', 20.0))
    safety_penalty_weight = float(reward_cfg.get('safety_penalty_weight', 2.0))

    if min_dist_to_obs < safety_distance:
        # 惩罚力度呈二次方关系，距离越近，惩罚急剧增大
        r -= ((1.0 - min_dist_to_obs / safety_distance) ** 2) * safety_penalty_weight
    
    # 4. 因果奖励塑造 - 反事实思考
    causal_credit_bonus = float(reward_cfg.get('causal_credit_bonus', 5.0))
    
    # 模拟默认动作（直行）的结果
    ego_prev = prev_obs['vehicle_info']['ego']
    default_action_speed = 0.3  # 默认中速直行
    
    # 简单的前向预测：如果上一步采取默认动作会怎样
    dt = curr_obs['test_setting']['dt']
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
    
    # 5. 世界模型增强的因果奖励
    if causal_reasoner is not None:
        try:
            # 使用世界模型进行更精确的因果分析
            import torch
            state_tensor = torch.FloatTensor(extract_state_causal(prev_obs, None, 3)).unsqueeze(0)
            action_tensor = torch.FloatTensor([0.3, 0.3]).unsqueeze(0)  # 默认动作
            
            # 计算反事实奖励
            counterfactual_contribution = causal_reasoner.counterfactual_reward(
                state_tensor, action_tensor, action_tensor, r
            )
            
            # 添加因果贡献奖励
            intervention_bonus = float(reward_cfg.get('intervention_bonus', 3.0))
            r += counterfactual_contribution * intervention_bonus
            
        except Exception as e:
            # 如果世界模型推理失败，继续使用基础奖励
            pass
        
    return r


def apply_safety_constraints_causal(action: np.ndarray, env, observation: dict) -> np.ndarray:
    """
    因果增强的安全约束函数
    """
    try:
        ego = observation['vehicle_info']['ego']
        ego_v = float(ego['v_mps'])
        
        # 获取车辆参数
        shape = ego['shape']
        rmin = float(shape.get('min_turn_radius', 14.2))
        width = float(shape.get('width', 6.7))
        
        # 获取动力学参数
        dynamics_params = env.controller.control_info.test_setting.get('dynamics_params', {})
        vmax = float(dynamics_params.get('max_speed', 6.0))
        a_max = float(dynamics_params.get('a_max', 2.5))
        b_max = float(dynamics_params.get('b_max', 3.5))
        omega_max = float(dynamics_params.get('omega_abs_max', 1.0))
        
        v_l, v_r = float(action[0]), float(action[1])
        
        # 1. 速度限制
        v_l = np.clip(v_l, -1.0, 1.0)
        v_r = np.clip(v_r, -1.0, 1.0)
        
        # 2. 最小转弯半径约束
        if rmin > 1e-6:
            current_speed = max(abs(v_l), abs(v_r)) * vmax
            if current_speed > 0.1:
                max_omega = current_speed / rmin
                max_omega = min(max_omega, omega_max)
                max_diff = max_omega * width / vmax
                
                diff = v_r - v_l
                if abs(diff) > max_diff:
                    scale = max_diff / (abs(diff) + 1e-6)
                    mid = 0.5 * (v_l + v_r)
                    half = 0.5 * abs(diff) * scale
                    v_l = mid - half * np.sign(diff)
                    v_r = mid + half * np.sign(diff)
        
        # 3. 加速度限制
        dt = 0.1
        max_accel = a_max * dt / vmax
        max_decel = b_max * dt / vmax
        
        v_l = np.clip(v_l, ego_v/vmax - max_decel, ego_v/vmax + max_accel)
        v_r = np.clip(v_r, ego_v/vmax - max_decel, ego_v/vmax + max_accel)
        
        # 4. 角速度限制
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
        
        return np.array([v_l, v_r], dtype=np.float32)
        
    except Exception as e:
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config_causal.yaml', help='YAML/JSON config for training')
    parser.add_argument('--logdir', type=str, default='runs', help='TensorBoard/CSV log dir')
    parser.add_argument('--tag', type=str, default='causal_default', help='run tag')
    args = parser.parse_args()
    
    # 固定随机种子
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

    # 读取训练配置
    cfg = load_config_file(args.cfg)
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    net_cfg = cfg.get('network', {}) if isinstance(cfg, dict) else {}
    opt_cfg = cfg.get('optimizer', {}) if isinstance(cfg, dict) else {}
    reward_cfg = cfg.get('reward', {}) if isinstance(cfg, dict) else {}
    causal_cfg = cfg.get('causal', {}) if isinstance(cfg, dict) else {}

    NUM_OBS = int(train_cfg.get('num_obstacles', 3))
    state_dim = 5 + 4 * NUM_OBS + 5  # 增加5维边界感知信息
    action_dim = 2
    hidden_dim = int(net_cfg.get('hidden_dim', 512))
    gamma = float(opt_cfg.get('gamma', 0.99))
    tau = float(opt_cfg.get('tau', 0.005))
    actor_lr = float(opt_cfg.get('actor_lr', 2e-4))
    critic_lr = float(opt_cfg.get('critic_lr', 2e-4))
    buffer_size = int(opt_cfg.get('replay_size', 3_000_000))

    target_update_interval = int(opt_cfg.get('target_update_interval', 1))
    sequence_length = int(train_cfg.get('sequence_length', 15))
    
    # 初始化智能体
    agent = LiquidSACAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        hidden_dim=hidden_dim,
        gamma=gamma, 
        tau=tau, 
        actor_lr=actor_lr, 
        critic_lr=critic_lr,
        replay_buffer_capacity=buffer_size, 
        target_update_interval=target_update_interval,
        sequence_length=sequence_length
    )

    # 初始化世界模型和因果推理器
    world_model = WorldModel(state_dim, action_dim, int(causal_cfg.get('world_model_hidden', 256)))
    causal_reasoner = CausalReasoner(world_model, agent.device)
    world_model_trainer = WorldModelTrainer(world_model, float(causal_cfg.get('world_model_lr', 1e-3)))

    max_episodes = int(train_cfg.get('max_episodes', 3000))
    max_steps = int(train_cfg.get('max_steps', 2000))
    batch_size = int(train_cfg.get('batch_size', 512))
    save_interval = int(train_cfg.get('save_interval', 50))
    warmup_steps = int(train_cfg.get('warmup_steps', 15000))

    steps_done = 0

    # 日志器
    run_name = f"{args.tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(args.logdir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if TB_AVAILABLE else None
    csv_path = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write('episode,step,return,critic_loss,actor_loss,alpha_loss,alpha,world_model_loss,ego_v,ego_yawrate\n')

    print("=" * 60)
    print("因果强化学习增强的SAC训练开始")
    print("=" * 60)
    print(f"状态维度: {state_dim}")
    print(f"网络宽度: {hidden_dim}")
    print(f"序列长度: {sequence_length}")
    print(f"经验池大小: {buffer_size}")
    print(f"世界模型隐藏层: {causal_cfg.get('world_model_hidden', 256)}")
    print("=" * 60)

    for ep in range(max_episodes):
        scenario = random.choice(so.scenario_list)
        observation, _ = env.make(scenario, collision_lookup, read_only=True, save_img_path='')

        prev_obs = observation
        state = extract_state_causal(observation, collision_lookup, NUM_OBS)
        ep_ret = 0.0
        
        # 初始化隐藏状态
        actor_hidden_state = None

        for step in range(max_steps):
            if steps_done < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
                actor_hidden_state = None
            else:
                # 使用液态神经网络选择动作
                action, actor_hidden_state = agent.select_action(state, actor_hidden_state, evaluate=False)
                action = apply_safety_constraints_causal(action, env, observation)
                
            next_obs = env.step_rl(tuple(action.tolist()), collision_lookup)
            reward = calculate_reward_causal_enhanced(next_obs, prev_obs, reward_cfg, causal_reasoner)
            done = next_obs['test_setting']['end'] != -1
            next_state = extract_state_causal(next_obs, collision_lookup, NUM_OBS)

            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done, actor_hidden_state)
            metrics = agent.update(batch_size)
            steps_done += 1

            state = next_state
            prev_obs = next_obs
            ep_ret += reward
            if done:
                break

        # 训练世界模型
        world_model_losses = None
        if steps_done > warmup_steps and len(agent.replay_buffer) > batch_size:
            world_model_losses = world_model_trainer.train_from_buffer(agent.replay_buffer, batch_size, epochs=1)

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
            
            if world_model_losses is not None:
                writer.add_scalar('loss/world_model', world_model_losses['total_loss'], ep)
                writer.add_scalar('loss/world_model_state', world_model_losses['state_loss'], ep)
                writer.add_scalar('loss/world_model_reward', world_model_losses['reward_loss'], ep)
        
        with open(csv_path, 'a') as f:
            world_model_loss_str = world_model_losses['total_loss'] if world_model_losses else ''
            f.write(f"{ep+1},{step+1},{ep_ret:.4f},{(metrics or {}).get('critic_loss','')},{(metrics or {}).get('actor_loss','')},{(metrics or {}).get('alpha_loss','')},{(metrics or {}).get('alpha','')},{world_model_loss_str},{ego_v:.4f},{ego_yawrate:.4f}\n")

        print(f"Episode {ep+1} | steps {step+1} | return {ep_ret:.2f} | world_model_loss: {world_model_losses['total_loss']:.4f}" if world_model_losses else f"Episode {ep+1} | steps {step+1} | return {ep_ret:.2f}")

        if (ep + 1) % save_interval == 0:
            model_dir = os.path.join(output_dir, 'causal_sac_model')
            os.makedirs(model_dir, exist_ok=True)
            import torch
            torch.save(agent.actor.state_dict(), os.path.join(model_dir, f'actor_ep_{ep+1}.pth'))
            torch.save(agent.critic.state_dict(), os.path.join(model_dir, f'critic_ep_{ep+1}.pth'))
            torch.save(world_model.state_dict(), os.path.join(model_dir, f'world_model_ep_{ep+1}.pth'))
            print(f"Saved Causal SAC models to {model_dir}")


if __name__ == '__main__':
    main()
