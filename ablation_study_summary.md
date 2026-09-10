# DCT 消融实验结果汇总

**日期**: 2026年9月9日  
**状态**: 所有旧消融实验数据已清理，需要重新训练

---

## ❌ 缺失的消融实验（需要重跑）

以下消融实验需要重新训练 50 epochs × 4 条件：

| 消融条件 | 描述 | 状态 |
|---------|------|------|
| NLL-only | 仅负对数似然损失 | ❌ 待训练 |
| IPCW-only | NLL + IPCW 排名损失 | ❌ 待训练 |
| Direction-only | NLL + 方向正则化 | ❌ 待训练 |
| Full | NLL + IPCW + Direction | ❌ 待训练 |

> 旧数据来源 (`backups/frozen_bug/`, `dct_v3.2_tgsr_optimization/`) 已全部删除。
