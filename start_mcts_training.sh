#!/bin/bash

# MCTS增强训练启动脚本
# 专门为MCTS规划优化的训练流程

echo "=========================================="
echo "启动MCTS增强智能体训练"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: Python未安装或不在PATH中"
    exit 1
fi

# 检查依赖包
echo "检查依赖包..."
python -c "import torch, ncps, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的依赖包"
    echo "请运行: pip install -r requirements_lnn.txt"
    exit 1
fi

# 检查GPU可用性
echo "检查GPU可用性..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
if [ $? -ne 0 ]; then
    echo "警告: 无法检查GPU状态"
fi

# 创建输出目录
mkdir -p demo/demo_outputs/mcts_sac_model
mkdir -p demo/demo_outputs/world_model
mkdir -p demo/demo_outputs/training_state
mkdir -p runs

# 设置训练参数
CONFIG_FILE="config_mcts.yaml"
TAG="mcts_training_$(date +%Y%m%d_%H%M%S)"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo "配置文件: $CONFIG_FILE"
echo "训练标签: $TAG"
echo "开始时间: $(date)"
echo "=========================================="

# 显示MCTS配置信息
echo "MCTS配置信息:"
echo "- 模拟次数: $(grep 'num_simulations:' $CONFIG_FILE | awk '{print $2}')"
echo "- 探索常数: $(grep 'exploration_constant:' $CONFIG_FILE | awk '{print $2}')"
echo "- 最大深度: $(grep 'max_depth:' $CONFIG_FILE | awk '{print $2}')"
echo "- 启用MCTS: $(grep 'enable_mcts:' $CONFIG_FILE | awk '{print $2}')"
echo "=========================================="

# 启动训练
echo "开始MCTS增强训练..."
python train_advanced.py --cfg "$CONFIG_FILE" --tag "$TAG"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "MCTS训练完成!"
    echo "结束时间: $(date)"
    echo "模型保存在: demo/demo_outputs/mcts_sac_model/"
    echo "训练日志保存在: runs/"
    echo "=========================================="
    
    # 显示训练统计信息
    echo "训练统计信息:"
    if [ -f "demo/demo_outputs/training_state/curriculum_state.json" ]; then
        echo "- 课程学习状态: 已保存"
    fi
    if [ -f "demo/demo_outputs/training_state/reward_statistics.json" ]; then
        echo "- 奖励统计信息: 已保存"
    fi
    
    echo "=========================================="
    echo "使用TensorBoard查看训练过程:"
    echo "tensorboard --logdir runs/"
    echo "=========================================="
else
    echo "=========================================="
    echo "MCTS训练失败!"
    echo "请检查错误信息并重试"
    echo "=========================================="
    exit 1
fi
