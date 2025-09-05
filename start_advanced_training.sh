#!/bin/bash

# 高级智能体训练启动脚本
# 集成四支柱优化技术的完整训练流程

echo "=========================================="
echo "启动高级智能体训练"
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

# 创建输出目录
mkdir -p demo/demo_outputs/advanced_sac_model
mkdir -p demo/demo_outputs/world_model
mkdir -p demo/demo_outputs/training_state
mkdir -p runs

# 设置训练参数
CONFIG_FILE="config_advanced.yaml"
TAG="advanced_training_$(date +%Y%m%d_%H%M%S)"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo "配置文件: $CONFIG_FILE"
echo "训练标签: $TAG"
echo "开始时间: $(date)"
echo "=========================================="

# 启动训练
python train_advanced.py --cfg "$CONFIG_FILE" --tag "$TAG"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练完成!"
    echo "结束时间: $(date)"
    echo "模型保存在: demo/demo_outputs/advanced_sac_model/"
    echo "训练日志保存在: runs/"
    echo "=========================================="
else
    echo "=========================================="
    echo "训练失败!"
    echo "请检查错误信息并重试"
    echo "=========================================="
    exit 1
fi
