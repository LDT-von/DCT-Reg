# DCT v3.13 实验接力文档（开新窗口专用）

> 写于 2026-09-30 14:13（UTC+8），主对话结束时交付。
> 目标：让"开新窗口"的我（或其他 agent）能 30 秒内定位环境与未完成工作，直接接着干。
>
> 重要边界：本文档只描述**接下来要做什么**，不接管任何已完成的实验数据，不覆盖任何已 commit 的结果。
> 已 commit 现状：`git log` 顶部 `d52e4bd → 4775d25 → 93d8314 → 5e8e44b`（均已推 `origin/main`）。

---

## 0. 一句话状态

- **实验（70 fold loss ablation）跑完了 ✅**，成绩已写进 `v311_vs_v313_uni_comparison.md` Appendix A 并 push。
- **实验代码本身还有大块未做**（按 `paper/V313_IMPLEMENTATION_PLAN.md`）：新配方权重解析、SlotSPE 原版适配、direct/independent 对照、outer_test 协议。
- **中文论文 Markdown 已起草**，Word 尚未生成。
- **本仓库不需要 re-clone、不需要重装环境**，开新窗口只要 `cd /data1/DCT-Reg` 就续。

---

## 1. 工作区与硬件（已确认）

| 项目 | 值 |
|---|---|
| 工作目录 | `/data1/DCT-Reg` |
| 当前分支 | `main`，与 `origin/main` 同步（HEAD `d52e4bd`） |
| 磁盘 | `/data1` 10T，已用 559G，剩余 8.9T；`/` 97G 已用 94%（仅 6.3G 余） |
| GPU | **2 × RTX 5090 (32 GB 各)**，`nvidia-smi` 可见，driver 570.86.10 |
| CUDA | `torch.version.cuda = 12.4`，`torch.cuda.is_available() = True` |
| PyTorch | `2.6.0+cu124` |
| Conda envs | `base`（默认激活）、`trisurv`（在 `/home/ubuntu/.conda/envs/trisurv`，计划中需要它跑 v3.13 CLI）、`vmkla`（无关） |
| 第三方案件 | `third_party/SlotSPE/` 已 clone（与原版源码比对过、已校验） |
| 数据集 | `/data1/dataset_csv/{kirc,blca,...}`（uni2h 五折 split 已经在 `configs/fixed_5fold/` 下） |
| 关键文件 | `configs/dct_v313_blca_uni2h.yaml`、`configs/dct_v313_<cancer>_uni.yaml`（9 个癌种）、`survot_rank/cli.py`、`paper/V313_IMPLEMENTATION_PLAN.md`、`paper/V313_EXPERIMENT_PLAN.md`、`paper/DCT_v313_初稿.md` |

> 注意：当前 `python3` 指向 `base` env（`/data/env/data/env/anaconda/bin/python3`）。这是 CLI 测试时遇到的；正式训练/调权重使用 `trisurv` env。

---

## 2. 主对话已确认的结论（不要重做）

1. **70-fold loss-component ablation 全完成**：KIRC + BLCA × Exp0..Exp6 × 5 fold = 70 jobs。
2. **FROZEN_ARGUMENT 静默覆盖已修复**（commit `93d8314`），否则 ablation 不会出现当前差异。
3. **Exp6 FULL 在两癌种都是 mean c-index 最高**：KIRC .8224 / BLCA .7238。
4. **详细数字 + epoch** 已在 `v311_vs_v313_uni_comparison.md` 附录 A。
5. **不得覆盖 `.ablation_buggy_runs_backup/`**（修复前 buggy 跑的备份，保留供回溯）。

---

## 3. 接下来要做的事（按 `V313_IMPLEMENTATION_PLAN.md` 的顺序）

### 3.1 优先级 P0（先做这几件）

| # | 任务 | 工作量 | 入口 / 命令 |
|---|---|---|---|
| 1 | **新配方权重解析**：`dct_v313_recon_weighting=per_branch`（关闭一支只置零该支，不再二次降权）；CLI 加 `--set dct_v313_recon_weighting=per_branch` 解析 | 改 `survot_rank/config.py` + `training/train_runner.py` 单支权重构造 + 加单元测试 | `survot_rank/training/train_runner.py:592 / :657` 是关键行 |
| 2 | **SlotSPE 原版适配器**：封装 `third_party/SlotSPE/models/`，隔离命名空间；两种配方 `matched`（lr=5e-4 / batch=8 / AdamW）与 `native`（lr=5e-4 / batch=32 / Adam / alpha_surv=0.5）；同一输入/划分/终点 | 新建 `survot_rank/research/slotspe_adapter.py` + `configs/slotspe_matched_<cancer>_uni2h.yaml` + `configs/slotspe_native_<cancer>_uni2h.yaml` |
| 3 | **普通重建对照 `dct_v313_cross_mode=transport\|direct`**：direct 不经 OT 直接 WSI slots → cross decoder；与 default transport 比 | 改 `model.py` 的 cross decoder memory 来源；新增开关 |
| 4 | **独立耦合对照 `dct_v313_plan_mode=learned\|independent`**：用相同边际的 `T=abᵀ` 替换学习计划（训练 + eval 都覆盖）；**不得**用 `1/(K_w K_o)` 替代 | 改 `model.py:156–181` 计划生成；加单元测试非负/行和等于边际 |
| 5 | **outer_test 协议**：原 val 改外层 test，原 train 内分 20% inner val，按事件/删失分层，split_seed=3；选模只在 inner val；不重训 | 改数据 loader + 加 `evaluation_protocol=outer_test\|legacy_val` 开关 |

### 3.2 调度的活

6. **统一调度入口**：CLI 选项 `--protocol --arm --cancer --fold --seed --gpu --jobs-per-gpu --execute`；默认仅打印计划；并发默认每 GPU 1 个训练子进程；GPU 映射在 import torch 前生效（避免两队列抢卡）
7. **任务目录命名**：`<version>_<protocol>_<arm>_<cancer>_seed<seed>/`，绝不与历史目录碰撞
8. **结果核验工具**：扫 `epoch_curve_fold*.csv` + 检查配置签名 + 标 completed/failed/interrupted（旧 log 的 "best cindex" 文本不是完成证明）

### 3.3 中文论文与 Word

9. 完善 `paper/DCT_v313_初稿.md`：摘要、引言、相关工作、方法、实验、讨论、结论
10. 生成 Word：`pandoc paper/DCT_v313_初稿.md -o paper/DCT_v313_初稿.docx --reference-doc=...`（参考旧 `paper/DCT_唯一初稿.docx` 版式）
11. 公式与图：原生可编辑公式（pandoc + LaTeX），框架图 = 预测路径 + 训练辅助 self/cross 支路（删 v3.10 的 direction 训练端点）

### 3.4 验收（每实现一段，先跑过这些再继续）

| 检查 | 标准 |
|---|---|
| 配置到实际目标 | CLI 覆盖经完整解析 + 构造后仍生效；七臂有效系数与实现表一致 |
| 单支重建 | per_branch 下，保留支路 = 0.05；关闭支路系数与梯度贡献 = 0 |
| direct 模式 | cross decoder 输入 memory 与 transport 不同但其他全等 |
| independent 计划 | 非负、总质量=1、行和=列边际 |
| 梯度路径 | transport cross 对运输路径保留梯度；direct 不通过 T 反传 |
| 原版适配 | 同种子下 logits 与辅助损失归一化与原版一致 |
| 数据身份 | train/val/test 无患者重叠；坏样本触发带 ID 的错误 |
| 重载一致性 | save→reload→同一 checkpoint 复现风险预测（容差内） |

---

## 4. 您（用户）需要开新窗口执行的最小命令集

```bash
# 4.1 进入工作区 + 激活正确的 conda env（不用 base env！）
cd /data1/DCT-Reg
source /home/ubuntu/.conda/etc/profile.d/conda.sh
conda activate trisurv

# 4.2 跑通现状自检（一行命令，确认仍在 commit d52e4bd + GPU 可见）
\
git log --oneline -1 && \
nvidia-smi --query-gpu=index,name --format=csv && \
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())"

# 4.3 跑一下 CLI 冒烟（不真训练）
PYTHONPATH=/data1/DCT-Reg \
  /home/ubuntu/.conda/envs/trisurv/bin/python -m survot_rank.cli doctor \
  --config configs/dct_v313_blca_uni2h.yaml

# 4.4 把当前结论文件 cat 出来确认（不必改）
sed -n '1,200p' v311_vs_v313_uni_comparison.md | head -80
```

**接下来开干**：让新窗口里的我读 `NEXT_STEPS.md` + `paper/V313_IMPLEMENTATION_PLAN.md`，先按 **§3.1 第 1 项**（per_branch 权重解析）开始写代码 + 单元测试。

---

## 5. 绝对不能动的（红线）

1. `results/dct_v313_ablation_*` 与 `logs/ablation_*`（70 fold 真实结果）—— **只读**。
2. `configs/fixed_5fold/`（五折 split）—— 任何新协议要在它之外另起 `fixed_5fold_outer_test_seed3/`。
3. `paper/DCT_唯一初稿.md` / `.docx`（v3.10 旧稿，**只读**），新稿写 `paper/DCT_v313_初稿.md`。
4. `third_party/SlotSPE/`（原版源码，**只读**）；适配器在 `survot_rank/research/slotspe_adapter.py` 内包装，不修改原版文件。
5. `.ablation_buggy_runs_backup/`（buggy 跑备份，**不删不 commit**）。

---

## 6. 已知 bug / 已修复但要警惕

- 修复 `93d8314` 前，所有 `dct_v313_*` 类的 `--set` 覆盖会被 `FROZEN_ARGUMENTS` 静默重置 —— **不要回退到该 commit 之前的代码跑 ablation**。
- CLI 默认跑 train/val，没 outer test；要做 outer_test 必须先把 §3.1 #5 实现好。
- 旧 `scripts/run_slotspe_blca_paper.sh` 用 UNI 1024 维 + 4096 patches + batch 32，与当前 uni2h (1536 + 2048 + 8) 不一致，不能当对照基线。

---

## 7. 一张"我下一步该做哪一行"的速查表

| 我现在想... | 该做什么 |
|---|---|
| 看现状 | `git log --oneline -5` + `nvidia-smi` |
| 跑 ablation 旧任务 | **别动**——已完成 |
| 加 per_branch 权重 | 改 `survot_rank/config.py` 与 `training/train_runner.py`，加测试在 `tests/test_recon_weighting_per_branch.py` |
| 跑 outer_test | 先实现 §3.1 #5，否则别跑 |
| 出图 / 论文表 | 先用 `epoch_curve_fold*.csv` + `paper/DCT_v313_初稿.md`；不要重新训练 |
| 写 Word | `pandoc paper/DCT_v313_初稿.md -o paper/DCT_v313_初稿.docx`（需先写完 .md） |
| 加癌种 | 5 个候选（BLCA/KIRC/HNSC/LUSC/SKCM）已有 `configs/dct_v313_<cancer>_uni.yaml`；uni2h 版本缺，需手动改 |

---

**主对话交付完成。如要继续，请新窗口贴这段："读 `/data1/DCT-Reg/NEXT_STEPS.md`，按 §3.1 顺序开干。"**
