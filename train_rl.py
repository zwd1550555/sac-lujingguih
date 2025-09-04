# -*- coding: utf-8 -*-
"""
Training entry for SAC on onsite_mine environment.
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


def extract_state(observation: dict, num_obstacles: int = 3) -> np.ndarray:
    ego = observation['vehicle_info']['ego']
    goal = observation['test_setting']['goal']

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
    return np.array(state, dtype=np.float32)


def calculate_reward(curr_obs: dict, prev_obs: dict, reward_cfg: dict) -> float:
    ego = curr_obs['vehicle_info']['ego']
    end = curr_obs['test_setting']['end']
    reach_goal_bonus = float(reward_cfg.get('reach_goal_bonus', 500.0))
    collision_penalty = float(reward_cfg.get('collision_penalty', -500.0))
    timeout_penalty = float(reward_cfg.get('timeout_penalty', -100.0))
    if end == 4:
        return reach_goal_bonus
    if end in (2, 3):
        return collision_penalty
    if end == 1:
        return timeout_penalty

    goal = curr_obs['test_setting']['goal']
    gx = float(np.mean(goal['x']))
    gy = float(np.mean(goal['y']))

    dist_now = math.hypot(float(ego['x']) - gx, float(ego['y']) - gy)
    ego_prev = prev_obs['vehicle_info']['ego']
    dist_prev = math.hypot(float(ego_prev['x']) - gx, float(ego_prev['y']) - gy)

    progress_weight = float(reward_cfg.get('progress_weight', 2.0))
    speed_weight = float(reward_cfg.get('speed_weight', 0.1))
    yawrate_penalty = float(reward_cfg.get('yawrate_penalty', 0.1))
    time_penalty = float(reward_cfg.get('time_penalty', 0.5))

    r = 0.0
    r += (dist_prev - dist_now) * progress_weight
    r += float(ego['v_mps']) * speed_weight
    r -= abs(float(ego['yawrate_radps'])) * yawrate_penalty
    r -= time_penalty
    return r


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
    state_dim = 5 + 4 * NUM_OBS
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

    max_episodes = int(train_cfg.get('max_episodes', 1000))
    max_steps = int(train_cfg.get('max_steps', 1000))
    batch_size = int(train_cfg.get('batch_size', 256))
    save_interval = int(train_cfg.get('save_interval', 50))
    warmup_steps = int(train_cfg.get('warmup_steps', 5000))

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

        prev_obs = observation
        state = extract_state(observation, NUM_OBS)
        ep_ret = 0.0

        for step in range(max_steps):
            if steps_done < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
            else:
                # 安全层：基于 R_min 对差速动作做软裁剪，避免极端转弯
                action = agent.select_action(state)
                try:
                    rmin = float(env.controller.observation.vehicle_info['ego']['shape'].get('min_turn_radius', 0.0) or 0.0)
                    if rmin > 1e-6:
                        # 将左右履带速度向差速更小的方向收缩（减小 |v_r - v_l|）
                        v_l, v_r = float(action[0]), float(action[1])
                        # 粗略按当前速度上界估计允许的差速比例
                        vmax = env.controller.control_info.test_setting.get('dynamics_params',{}).get('max_speed',5.0)
                        # 允许的最大角速度 ~ v / rmin，映射到差速：|v_r - v_l| <= omega_max * wheel_base
                        wb = env.controller.observation.vehicle_info['ego']['shape']['width']
                        omega_max = vmax / max(rmin, 1e-6)
                        diff_max = omega_max * wb / max(vmax, 1e-6)
                        diff = v_r - v_l
                        if abs(diff) > diff_max:
                            scale = diff_max / (abs(diff) + 1e-6)
                            mid = 0.5 * (v_l + v_r)
                            half = 0.5 * abs(diff) * scale
                            v_l_new = mid - half * np.sign(diff)
                            v_r_new = mid + half * np.sign(diff)
                            action = np.array([v_l_new, v_r_new], dtype=np.float32)
                        action = np.clip(action, -1.0, 1.0)
                except Exception:
                    pass
            next_obs = env.step_rl(tuple(action.tolist()), collision_lookup)
            reward = calculate_reward(next_obs, prev_obs, reward_cfg)
            done = next_obs['test_setting']['end'] != -1
            next_state = extract_state(next_obs, NUM_OBS)

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


