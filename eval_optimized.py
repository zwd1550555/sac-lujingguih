#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的SAC评估脚本
用于测试训练好的履带车路径规划与动态避障模型

功能特性:
- 自动查找最新的训练模型
- 支持多种评估配置
- 提供详细的评估报告
- 支持可视化和图片保存

使用方法:
python eval_optimized.py --cfg config_optimized.yaml --actor demo/demo_outputs/sac_model/actor_ep_2000.pth
python eval_optimized.py --cfg config_causal.yaml --episodes 20 --no_vis
"""

import os
import sys
import subprocess
import argparse
import glob
from datetime import datetime

def find_latest_model(model_dir: str) -> str:
    """
    自动查找指定目录下最新的模型文件
    
    Args:
        model_dir (str): 模型文件所在目录路径
        
    Returns:
        str: 最新模型文件的完整路径，如果未找到则返回空字符串
        
    功能说明:
        - 扫描目录中所有以 'actor_ep_' 开头的 .pth 文件
        - 根据文件名中的episode编号进行排序
        - 返回episode编号最大的模型文件
    """
    # 检查目录是否存在
    if not os.path.exists(model_dir):
        return ""
    
    # 使用glob模式匹配所有actor模型文件
    actor_files = glob.glob(os.path.join(model_dir, "actor_ep_*.pth"))
    if not actor_files:
        return ""
    
    # 按文件名中的episode数排序（提取episode编号进行数值排序）
    # 例如: actor_ep_100.pth -> 100, actor_ep_2000.pth -> 2000
    actor_files.sort(key=lambda x: int(x.split('_ep_')[1].split('.')[0]))
    
    # 返回episode编号最大的文件（最新的模型）
    return actor_files[-1]

def main():
    """
    主函数：解析命令行参数并执行模型评估
    
    功能流程:
    1. 解析命令行参数
    2. 验证配置文件和模型文件
    3. 构建评估命令
    4. 执行评估并处理结果
    
    Returns:
        int: 程序退出码，0表示成功，1表示失败
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='优化的SAC评估脚本')
    parser.add_argument('--cfg', type=str, default='config_optimized.yaml', 
                       help='配置文件路径 (YAML格式，包含训练和网络配置)')
    parser.add_argument('--actor', type=str, default='', 
                       help='Actor模型文件路径 (PyTorch .pth文件)')
    parser.add_argument('--model_dir', type=str, default='demo/demo_outputs/sac_model', 
                       help='模型目录路径，用于自动查找最新模型')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='评估轮数 (每个episode运行一个完整场景)')
    parser.add_argument('--no_vis', action='store_true', 
                       help='关闭可视化显示 (提高评估速度)')
    parser.add_argument('--save_img_path', type=str, default='', 
                       help='保存评估过程图片的路径 (可选)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 验证配置文件是否存在
    if not os.path.exists(args.cfg):
        print(f"错误: 配置文件 {args.cfg} 不存在")
        print("请确保配置文件路径正确，或使用默认的 config_optimized.yaml")
        return 1
    
    # 确定要使用的模型文件路径
    actor_path = args.actor
    if not actor_path:
        # 如果未指定具体模型文件，则自动查找最新模型
        actor_path = find_latest_model(args.model_dir)
        if not actor_path:
            print(f"错误: 在 {args.model_dir} 中未找到模型文件")
            print("请使用 --actor 参数指定模型路径，或确保模型目录中有训练好的模型")
            return 1
    
    # 验证模型文件是否存在
    if not os.path.exists(actor_path):
        print(f"错误: 模型文件 {actor_path} 不存在")
        print("请检查模型文件路径是否正确")
        return 1
    
    # 构建评估命令
    # 调用原始的 eval_rl.py 脚本进行实际评估
    cmd = [
        'python', 'eval_rl.py',           # 调用原始评估脚本
        '--cfg', args.cfg,                # 传递配置文件
        '--actor', actor_path,            # 传递模型文件路径
        '--episodes', str(args.episodes)  # 传递评估轮数
    ]
    
    # 根据参数添加可选选项
    if args.no_vis:
        cmd.append('--no_vis')  # 关闭可视化
    
    if args.save_img_path:
        cmd.extend(['--save_img_path', args.save_img_path])  # 设置图片保存路径
    
    # 显示评估信息
    print("=" * 60)
    print("开始模型评估")
    print("=" * 60)
    print(f"配置文件: {args.cfg}")
    print(f"模型文件: {actor_path}")
    print(f"评估轮数: {args.episodes}")
    print(f"可视化: {'关闭' if args.no_vis else '开启'}")
    if args.save_img_path:
        print(f"图片保存路径: {args.save_img_path}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 执行评估命令
        # subprocess.run 用于调用外部Python脚本
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        # 评估成功完成
        print("=" * 60)
        print("评估完成!")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        # 评估过程中出现错误
        print(f"评估失败: {e}")
        print("请检查配置文件格式和模型文件完整性")
        return 1
    except KeyboardInterrupt:
        # 用户手动中断评估
        print("\n评估被用户中断")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
