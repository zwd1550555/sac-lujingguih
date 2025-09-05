#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的SAC训练脚本
用于履带车快速路径规划与动态避障

使用方法:
python train_optimized.py --cfg config_optimized.yaml --tag optimized_run
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='优化的SAC训练脚本')
    parser.add_argument('--cfg', type=str, default='config_optimized.yaml', 
                       help='配置文件路径')
    parser.add_argument('--tag', type=str, default='optimized', 
                       help='训练标签')
    parser.add_argument('--logdir', type=str, default='runs', 
                       help='日志目录')
    parser.add_argument('--gpu', type=int, default=0, 
                       help='GPU设备ID')
    parser.add_argument('--resume', type=str, default='', 
                       help='恢复训练的模型路径')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.cfg):
        print(f"错误: 配置文件 {args.cfg} 不存在")
        print("请确保配置文件存在，或使用默认的 config_optimized.yaml")
        return
    
    # 设置环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # 构建训练命令
    cmd = [
        'python', 'train_rl.py',
        '--cfg', args.cfg,
        '--tag', args.tag,
        '--logdir', args.logdir
    ]
    
    print("=" * 60)
    print("开始优化的SAC训练")
    print("=" * 60)
    print(f"配置文件: {args.cfg}")
    print(f"训练标签: {args.tag}")
    print(f"日志目录: {args.logdir}")
    print(f"GPU设备: {args.gpu}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 运行训练
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("=" * 60)
        print("训练完成!")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
    except subprocess.CalledProcessError as e:
        print(f"训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
