#!/bin/bash

# 监控消融实验进度

cd /data1/DCT-Reg

echo "========================================================"
echo "DCT v3.10 消融实验进度监控"
echo "========================================================"
echo ""

# 检查正在运行的进程
echo "🏃 正在运行的实验:"
ps aux | grep -E "survot_rank.*train" | grep -v grep | wc -l | xargs -I {} echo "  {} 个进程"
echo ""

# 检查GPU使用
echo "🎮 GPU状态:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1
echo ""

# 统计各变体完成情况
echo "📊 各变体完成情况:"
echo ""

for variant in nll_only ipcw_only direction_only; do
    echo "【${variant}】"
    
    # 统计CSV文件
    csv_count=$(find results/dct_v3.10_experiments/robust/${variant} -name "epoch_curve_fold*.csv" 2>/dev/null | wc -l)
    pkl_count=$(find results/dct_v3.10_experiments/robust/${variant} -name "split_*_results_final.pkl" 2>/dev/null | wc -l)
    
    echo "  训练日志文件: ${csv_count}/5"
    echo "  最终结果文件: ${pkl_count}/5"
    
    # 显示最新的训练epoch
    latest_csv=$(find results/dct_v3.10_experiments/robust/${variant} -name "epoch_curve_fold*.csv" 2>/dev/null | sort -t'fold' -k2 -n | tail -1)
    
    if [ -n "$latest_csv" ]; then
        fold=$(echo "$latest_csv" | grep -oP 'fold\K[0-9]+')
        epochs=$(wc -l < "$latest_csv" 2>/dev/null || echo "0")
        epochs=$((epochs - 1))  # 减去header行
        
        if [ "$epochs" -gt 0 ]; then
            # 读取最佳C-index
            best_cindex=$(python3 -c "
import pandas as pd
try:
    df = pd.read_csv('$latest_csv')
    if 'val_cindex' in df.columns:
        print(f'{df[\"val_cindex\"].max():.4f}')
    else:
        print('N/A')
except:
    print('N/A')
" 2>/dev/null)
            
            echo "  最新进度: Fold ${fold} - Epoch ${epochs}/30 (C-index: ${best_cindex})"
        fi
    fi
    
    echo ""
done

echo "========================================================"
echo "💡 提示:"
echo "  - 总计需要完成: 15 个任务 (3个变体 × 5 folds)"
echo "  - 每个任务约需: 1-1.5小时"
echo "  - 预计总时间: ~22.5小时"
echo ""
echo "  运行此脚本查看进度:"
echo "  bash monitor_ablations.sh"
echo "========================================================"
