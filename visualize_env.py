#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于真实训练环境的可视化脚本（实时显示，不保存）

- 使用 dynamic_scenes.Env + ScenarioOrganizer 读取真实场景
- 动态障碍物（背景车辆）按时间步更新与绘制
- 实时可视化：开启 plt.ion()，不保存 PNG/GIF

用法：
  python visualize_env.py --inputs demo/demo_inputs --outputs demo/demo_outputs --max_scenarios 1 --max_steps 200

注意：
- 需要可用的图形后端（本地或WSL需 X Server）。若无 GUI，将无法弹窗。
"""

import os
import sys
import time
import argparse

import numpy as np

from dynamic_scenes.env import Env
from dynamic_scenes.scenarioOrganizer1 import ScenarioOrganizer
from dynamic_scenes.lookup import CollisionLookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, default='demo/demo_inputs', help='场景输入根目录，需包含 Scenarios/ 与 Maps/')
    parser.add_argument('--outputs', type=str, default='demo/demo_outputs', help='输出目录（仅供内部组件初始化，不会写文件）')
    parser.add_argument('--max_scenarios', type=int, default=1, help='最多可视化多少个场景')
    parser.add_argument('--max_steps', type=int, default=300, help='每个场景最多可视化多少步')
    parser.add_argument('--dt_scale', type=float, default=1.0, help='时间步播放倍率（>1 更快）')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    input_dir = args.inputs if os.path.isabs(args.inputs) else os.path.join(project_root, args.inputs)
    output_dir = args.outputs if os.path.isabs(args.outputs) else os.path.join(project_root, args.outputs)

    if not os.path.isdir(os.path.join(input_dir, 'Scenarios')):
        print(f"未找到场景，请将场景放入 {input_dir}/Scenarios 下")
        return

    print('初始化碰撞查找表...')
    collision_lookup = CollisionLookup()
    print('完成')

    env = Env()
    so = ScenarioOrganizer()
    so.load(input_dir, output_dir)
    if not so.scenario_list:
        print("未找到可视化场景")
        return

    scenarios_done = 0
    while scenarios_done < args.max_scenarios:
        scenario = so.next()
        if scenario is None:
            break

        # 打开实时显示，不保存
        scenario['test_settings']['visualize'] = True
        scenario['test_settings']['save_fig_whitout_show'] = False

        observation, traj = env.make(scenario, collision_lookup, read_only=True, save_img_path='')

        obs_last = observation
        step_count = 0
        # 若 test_conf 设定 dt，可用它控制播放节奏
        dt_local = float(observation['test_setting'].get('dt', 0.1)) if isinstance(observation, dict) else 0.1
        sleep_dt = max(0.0, dt_local / max(args.dt_scale, 1e-6))

        # 简单策略：保持直行（归一化轮速 0.3, 0.3）
        default_action = (0.3, 0.3)

        while observation['test_setting']['end'] == -1 and step_count < args.max_steps:
            # 触发可视化刷新：需要提供 observation_last 与 traj，traj_future 可传空字典
            next_obs = env.step(default_action, traj_future={}, observation_last=obs_last, traj=traj, collision_lookup=collision_lookup)
            obs_last = observation
            observation = next_obs
            step_count += 1
            # 控制刷新节奏
            if sleep_dt > 0:
                time.sleep(sleep_dt)

        scenarios_done += 1

    print('可视化完成')


if __name__ == '__main__':
    main()


