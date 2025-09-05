#!/bin/bash
# 快速启动脚本 - 履带车SAC训练优化

echo "=========================================="
echo "履带车SAC训练优化 - 快速启动"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查必要的包
echo "检查依赖包..."
python -c "import torch, numpy, yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "错误: 缺少必要的Python包 (torch, numpy, yaml)"
    echo "请安装: pip install torch numpy pyyaml"
    exit 1
fi

# 检查场景文件
if [ ! -d "demo/demo_inputs/Scenarios" ]; then
    echo "警告: 未找到训练场景目录 demo/demo_inputs/Scenarios"
    echo "请确保场景文件已正确放置"
fi

# 创建必要的目录
mkdir -p runs
mkdir -p demo/demo_outputs/sac_model

echo "=========================================="
echo "开始训练..."
echo "=========================================="

# 启动训练
python train_optimized.py --cfg config_optimized.yaml --tag optimized_$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "模型保存在: demo/demo_outputs/sac_model/"
echo "日志保存在: runs/"
echo ""
echo "评估模型请运行:"
echo "python eval_optimized.py --cfg config_optimized.yaml"
echo "=========================================="
