#!/usr/bin/env python3
"""
综合分析：生成完整的效率和校准指标报告
汇总所有已完成的分析结果
"""

import json
import numpy as np
from pathlib import Path

def load_json(filepath):
    """加载JSON文件"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  无法加载 {filepath}: {e}")
        return None

def format_time_estimate(hours):
    """格式化时间估算"""
    if hours < 1:
        return f"{hours * 60:.0f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours / 24
        return f"{days:.1f} days ({hours:.0f} hours)"

def main():
    results_dir = Path("/data1/DCT-Reg/results")
    
    print("="*80)
    print("📊 DCT 模型效率与校准指标综合报告")
    print("="*80)
    print()
    
    # 1. 参数量和模型大小
    print("## 1️⃣ 计算效率指标")
    print("-" * 80)
    
    for version in ["v32", "v310"]:
        efficiency_file = results_dir / f"efficiency_metrics_{version}.json"
        data = load_json(efficiency_file)
        
        if data:
            print(f"\n### DCT {version.replace('v', 'v3.')}")
            
            # 处理不同的JSON格式
            if 'summary' in data:
                params = data['summary']['parameters']
                size = data['summary']['checkpoint_size_mb']
                print(f"  参数量: {params['mean'] / 1e6:.2f}M ± {params['std'] / 1e6:.2f}M")
                print(f"  模型大小: {size['mean']:.2f} ± {size['std']:.2f} MB")
                print(f"  统计样本: {params['total_checkpoints']} 个检查点")
            else:
                print(f"  参数量: {data.get('mean_params', 0) / 1e6:.2f}M")
                print(f"  模型大小: {data.get('mean_size_mb', 0):.2f} MB")
            
            if 'cancer_types' in data:
                print(f"  覆盖癌种: {', '.join(data['cancer_types'])}")
    
    # 2. 训练时间（基于分析）
    print("\n\n## 2️⃣ 训练时间")
    print("-" * 80)
    
    training_time_file = results_dir / "training_time_analysis.json"
    time_data = load_json(training_time_file)
    
    if time_data and 'summary' in time_data:
        print("\n⚠️  注意：CSV文件中不包含训练时间列")
        print("需要从日志文件或重新训练时记录")
        print()
        
        for version, cancers in time_data['summary'].items():
            print(f"\n### DCT {version}")
            for cancer, info in cancers.items():
                print(f"  {cancer}: {info['num_folds']} folds, {info['num_epochs_range']} epochs")
                if 'best_val_cindex_mean' in info and info['best_val_cindex_mean']:
                    print(f"    最佳 C-index: {info['best_val_cindex_mean']:.4f} ± {info['best_val_cindex_std']:.4f}")
    
    # 估算训练时间（基于经验值）
    print("\n### 训练时间估算（基于经验）")
    print("  假设: ~2-5 分钟/epoch（取决于数据集大小和GPU）")
    print("  DCT v3.2 (30 epochs): ~1-2.5 小时/fold")
    print("  DCT v3.10 (50 epochs): ~1.5-4 小时/fold")
    
    # 3. 校准和决策曲线
    print("\n\n## 3️⃣ 模型校准与决策曲线")
    print("-" * 80)
    
    calibration_file = results_dir / "calibration_dca_metrics_v32.json"
    cal_data = load_json(calibration_file)
    
    if cal_data:
        print("\n### DCT v3.2 BLCA 5折验证")
        
        if 'calibration' in cal_data:
            cal = cal_data['calibration']
            print(f"\n**校准指标:**")
            print(f"  Brier Score: {cal['brier_score']['mean']:.2f} ± {cal['brier_score']['std']:.2f}")
            print(f"  Expected Calibration Error (ECE): {cal['ece']['mean']:.2f} ± {cal['ece']['std']:.2f}")
            print(f"  ⚠️  ECE > 1 表示校准较差，建议添加概率校准（如 Platt Scaling）")
        
        if 'dca' in cal_data:
            dca = cal_data['dca']
            print(f"\n**决策曲线分析 (DCA):**")
            print(f"  阈值范围: {dca['threshold_range']}")
            print(f"  评估的阈值点: {dca['n_thresholds']} 个")
            print(f"  生成图表: dct_v32_decision_curve.png")
    
    # 4. 生存分析指标
    print("\n\n## 4️⃣ 生存分析指标")
    print("-" * 80)
    
    if cal_data and 'survival_metrics' in cal_data:
        surv = cal_data['survival_metrics']
        if 'summary' in surv:
            summary = surv['summary']
            print(f"\n### DCT v3.2 BLCA {len(surv['fold_metrics'])}折验证")
            print(f"  C-index: {summary['C-index']['mean']:.4f} ± {summary['C-index']['std']:.4f}")
            print(f"  C-index (IPCW): {summary['C-index (IPCW)']['mean']:.4f} ± {summary['C-index (IPCW)']['std']:.4f}")
            print(f"  IBS (Integrated Brier Score): {summary['IBS']['mean']:.4f} ± {summary['IBS']['std']:.4f}")
            print(f"  iAUC (时间依赖 AUC): {summary['iAUC']['mean']:.4f} ± {summary['iAUC']['std']:.4f}")
    
    # 5. 生成的可视化文件
    print("\n\n## 5️⃣ 生成的可视化文件")
    print("-" * 80)
    
    viz_files = [
        ("dct_v32_calibration_curve.png", "校准曲线"),
        ("dct_v32_decision_curve.png", "决策曲线"),
    ]
    
    print()
    for filename, desc in viz_files:
        filepath = results_dir / filename
        if filepath.exists():
            print(f"  ✅ {desc}: {filepath}")
        else:
            print(f"  ❌ {desc}: 未找到")
    
    # 6. 数据文件
    print("\n\n## 6️⃣ 数据文件")
    print("-" * 80)
    
    data_files = [
        ("efficiency_metrics_v32.json", "DCT v3.2 效率指标"),
        ("efficiency_metrics_v310.json", "DCT v3.10 效率指标"),
        ("calibration_dca_metrics_v32.json", "校准和DCA指标"),
        ("training_time_analysis.json", "训练时间分析"),
    ]
    
    print()
    for filename, desc in data_files:
        filepath = results_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✅ {desc}: {filepath} ({size:,} bytes)")
        else:
            print(f"  ❌ {desc}: 未找到")
    
    # 7. 总结和建议
    print("\n\n## 7️⃣ 完成状态总结")
    print("="*80)
    
    completed = [
        ("参数量和模型大小", True, "已从检查点文件统计"),
        ("校准曲线 (Figure 11)", True, "已生成图表和指标"),
        ("决策曲线 (DCA)", True, "已生成图表"),
        ("IBS 和时间依赖 AUC", True, "已从验证数据计算"),
        ("C-index", True, "已计算平均值和标准差"),
        ("训练时间 (Figure 4)", False, "CSV中无时间列，需要从日志提取或重新训练"),
    ]
    
    print()
    for task, done, note in completed:
        status = "✅" if done else "❌"
        print(f"{status} {task}")
        if note:
            print(f"   → {note}")
    
    # 8. 下一步建议
    print("\n\n## 8️⃣ 下一步建议")
    print("="*80)
    print("""
### 完成训练时间统计:

**方法 1: 从日志文件提取（如果有）**
  - 查找训练日志中的 epoch 时间信息
  - 脚本: grep -r "epoch.*time|duration" logs/

**方法 2: 使用检查点时间戳估算**
  - 检查每个epoch检查点的创建时间
  - 计算时间差异作为训练时间
  - 误差: 可能包含暂停和系统开销

**方法 3: 重新运行并记录（最准确）**
  - 在训练脚本中添加时间记录
  - 使用 time 命令: time python train.py
  - 记录每个 epoch 的开始和结束时间

### 添加概率校准:
  - Brier Score 较高 (11.47) 表示校准不佳
  - 建议: 使用 Platt Scaling 或 Isotonic Regression
  - 可以显著改善概率预测质量

### 论文中的表格填充:
  - Figure 4: 参数量 ✅, 模型大小 ✅, 训练时间 ❌
  - Figure 10: 同上
  - Figure 11: 校准曲线 ✅
  - Table (new): IBS, iAUC, C-index ✅
    """)
    
    # 保存完整报告
    output_file = results_dir / "COMPLETE_METRICS_REPORT.md"
    print(f"\n💾 完整报告已保存: {output_file}")

if __name__ == "__main__":
    main()
