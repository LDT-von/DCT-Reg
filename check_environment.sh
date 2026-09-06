#!/bin/bash
# 环境检查脚本 - 在运行实验前验证所有依赖

echo "=========================================="
echo "DCT-Reg 环境检查"
echo "=========================================="

cd /data1/DCT-Reg

# 1. 检查 Python
echo ""
echo "1. Python 环境"
if command -v python &> /dev/null; then
    echo "✓ Python: $(python --version)"
else
    echo "✗ Python 未找到"
    exit 1
fi

# 2. 检查关键 Python 包
echo ""
echo "2. Python 包"
python -c "
import sys
import importlib

packages = ['torch', 'pandas', 'numpy', 'yaml', 'tqdm']
missing = []

for pkg in packages:
    try:
        mod = importlib.import_module(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {pkg}: {version}')
    except ImportError:
        print(f'✗ {pkg}: 未安装')
        missing.append(pkg)

if missing:
    print(f'\\n缺少包: {missing}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "✗ 缺少必要的 Python 包"
    exit 1
fi

# 3. 检查 CUDA
echo ""
echo "3. CUDA"
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA 可用: {torch.cuda.get_device_name(0)}')
    print(f'  设备数量: {torch.cuda.device_count()}')
else:
    print('⚠️  CUDA 不可用，将使用 CPU')
"

# 4. 检查数据路径
echo ""
echo "4. 数据路径"

DATA_CSV="/data1/DCT-Reg/data/dataset_csv"
DATA_WSI="/data1/TCGA-UNI2-h-features"
CONFIG="configs/dct_v310_directional_regularized_transport.yaml"

if [ -d "$DATA_CSV" ]; then
    echo "✓ CSV 数据: $DATA_CSV"
else
    echo "✗ CSV 数据未找到: $DATA_CSV"
    exit 1
fi

if [ -d "$DATA_WSI" ]; then
    echo "✓ WSI 特征: $DATA_WSI"
else
    echo "⚠️  WSI 特征未找到: $DATA_WSI (可能会报错)"
fi

if [ -f "$CONFIG" ]; then
    echo "✓ 配置文件: $CONFIG"
else
    echo "✗ 配置文件未找到: $CONFIG"
    exit 1
fi

# 5. 检查检查点
echo ""
echo "5. 检查点"

CHECKPOINT_DIR="results/backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✓ 检查点目录: $CHECKPOINT_DIR"
    
    # 统计可用的 fold
    AVAILABLE_FOLDS=0
    for fold in 0 1 2 3 4; do
        if [ -f "$CHECKPOINT_DIR/model_best_s${fold}.pth" ]; then
            AVAILABLE_FOLDS=$((AVAILABLE_FOLDS + 1))
        fi
    done
    
    echo "  可用 folds: $AVAILABLE_FOLDS / 5"
    
    if [ $AVAILABLE_FOLDS -eq 0 ]; then
        echo "✗ 没有找到任何检查点文件"
        exit 1
    fi
else
    echo "✗ 检查点目录未找到"
    exit 1
fi

# 6. 检查脚本
echo ""
echo "6. 实验脚本"

SCRIPTS=(
    "scripts/run_small_gate_test.py"
    "scripts/ablate_reader.py"
    "scripts/e4_continuous_intervention_audit_v2.py"
    "scripts/run_reader_ablation_experiments.sh"
    "scripts/run_e4_direction_audit.sh"
    "run_all_experiments.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ] || [[ "$script" == *.py ]]; then
            echo "✓ $script"
        else
            echo "⚠️  $script (不可执行)"
        fi
    else
        echo "✗ $script (未找到)"
        exit 1
    fi
done

# 7. 检查输出目录权限
echo ""
echo "7. 输出目录"

mkdir -p results/small_test results/reader_ablation results/e4_audit_direction_only 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ 输出目录可写"
else
    echo "✗ 无法创建输出目录"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ 环境检查通过"
echo "=========================================="
echo ""
echo "可以运行实验:"
echo "  bash quick_test.sh          # 快速测试"
echo "  bash run_all_experiments.sh # 完整流程"
echo ""
