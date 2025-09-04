# -*- coding: utf-8 -*-
"""
Evaluation script for SAC agent.
Loads actor weights, runs episodes with visualize enabled (optional), logs metrics and optionally saves frames.
"""

import os
import math
import argparse
import numpy as np

from dynamic_scenes.env import Env
from dynamic_scenes.scenarioOrganizer1 import ScenarioOrganizer
from dynamic_scenes.lookup import CollisionLookup
from agent import SACAgent
from common.config_loader import load_config_file


def run_eval(cfg_path: str, actor_path: str, visualize: bool = True, episodes: int = 5, save_img_path: str = ''):
    cfg = load_config_file(cfg_path)
    train_cfg = cfg.get('train', {}) if isinstance(cfg, dict) else {}
    NUM_OBS = int(train_cfg.get('num_obstacles', 3))

    env = Env()
    so = ScenarioOrganizer()
    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(project_root, 'demo', 'demo_inputs')
    output_dir = os.path.join(project_root, 'demo', 'demo_outputs')
    so.load(input_dir, output_dir)
    if not so.scenario_list:
        print('No scenarios found.')
        return

    collision_lookup = CollisionLookup()

    # 根据训练网络结构构造 agent 并加载 actor
    net_cfg = cfg.get('network', {}) if isinstance(cfg, dict) else {}
    state_dim = 5 + 4 * NUM_OBS
    action_dim = 2
    hidden_dim = int(net_cfg.get('hidden_dim', 256))
    agent = SACAgent(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    import torch
    agent.actor.load_state_dict(torch.load(actor_path, map_location='cpu'))
    agent.actor.eval()

    returns = []
    collisions = 0
    out_of_bounds = 0
    for ep in range(episodes):
        scenario = so.next() or so.scenario_list[0]
        # 强制打开可视化
        scenario['test_settings']['visualize'] = visualize
        obs, _ = env.make(scenario, collision_lookup, read_only=True, save_img_path=save_img_path)
        ret = 0.0
        last_obs = obs
        steps = 0
        while True:
            # 构造状态
            ego = obs['vehicle_info']['ego']
            goal = obs['test_setting']['goal'] if 'test_setting' in obs else obs['test_setting']
            goal_xc = float(np.mean(goal['x']))
            goal_yc = float(np.mean(goal['y']))
            dx = goal_xc - float(ego['x'])
            dy = goal_yc - float(ego['y'])
            yaw = float(ego['yaw_rad'])
            gx_local = dx * math.cos(yaw) + dy * math.sin(yaw)
            gy_local = -dx * math.sin(yaw) + dy * math.cos(yaw)
            gdist = math.hypot(dx, dy)
            state = [float(ego['v_mps'])/10.0, float(ego['yawrate_radps'])/math.pi, gdist/100.0, gx_local/100.0, gy_local/100.0]
            others = []
            for vid, vinfo in obs['vehicle_info'].items():
                if vid == 'ego':
                    continue
                d = math.hypot(float(vinfo['x'])-float(ego['x']), float(vinfo['y'])-float(ego['y']))
                others.append((d, vinfo))
            others.sort(key=lambda x: x[0])
            for i in range(NUM_OBS):
                if i < len(others):
                    vinfo = others[i][1]
                    state.extend([
                        (float(vinfo['x'])-float(ego['x']))/50.0,
                        (float(vinfo['y'])-float(ego['y']))/50.0,
                        (float(vinfo['v_mps'])*math.cos(float(vinfo['yaw_rad']))-float(ego['v_mps'])*math.cos(yaw))/10.0,
                        (float(vinfo['v_mps'])*math.sin(float(vinfo['yaw_rad']))-float(ego['v_mps'])*math.sin(yaw))/10.0,
                    ])
                else:
                    state.extend([5.0,5.0,0.0,0.0])
            state = np.array(state, dtype=np.float32)

            action = agent.select_action(state, evaluate=True)
            obs = env.step_rl(tuple(action.tolist()), collision_lookup)
            ret += 0.0  # 评估阶段可按需重用训练奖励或单独定义
            steps += 1
            if obs['test_setting']['end'] != -1:
                if obs['test_setting']['end'] == 2:
                    collisions += 1
                if obs['test_setting']['end'] == 3:
                    out_of_bounds += 1
                break
        returns.append(ret)
        print(f"Eval Episode {ep+1}: steps={steps}")

    print(f"Avg Return: {np.mean(returns):.4f} | Collisions: {collisions} | OutOfBounds: {out_of_bounds}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--actor', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--no_vis', action='store_true')
    parser.add_argument('--save_img_path', type=str, default='')
    args = parser.parse_args()
    run_eval(args.cfg, args.actor, visualize=not args.no_vis, episodes=args.episodes, save_img_path=args.save_img_path)



