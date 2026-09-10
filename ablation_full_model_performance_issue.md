# Full 模型性能问题分析

**日期**: 2026年9月9日  
**状态**: 已清理所有旧消融实验数据

---

## ✅ 已清理的数据

以下目录已删除：
- `results/backups/direction_only_frozen_bug_20260903_173509`
- `results/backups/ipcw_only_frozen_bug_20260903_173509`
- `results/backups/nll_only_frozen_bug_20260903_173509`
- `results/dct_v3.2_tgsr_optimization/`

---

## ✅ 主表数据来源

主表 C-index 来自 `results/dct_v3.10/robust/final_50ep_old/` 的 `split_*_results_final.pkl` 文件：

| 癌种 | 均值 ± 标准差 | 来源 |
|------|------------|------|
| BLCA | 0.721 ± 0.016 | pkl |
| HNSC | 0.647 ± 0.071 | pkl |
| KIRC | 0.858 ± 0.012 | pkl |
| LUSC | 0.631 ± 0.055 | pkl |
| SKCM | 0.656 ± 0.047 | pkl |

使用 `scripts/extract_ablation_results.py` 可重新提取。

---

## ❌ 缺失的消融实验

真正的 NLL-only / IPCW-only / Direction-only 消融需要重新训练。
旧数据全部不可用。
