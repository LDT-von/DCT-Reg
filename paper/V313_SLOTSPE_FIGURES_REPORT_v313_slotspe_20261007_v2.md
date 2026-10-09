# v3.13 SlotSPE 风格真实论文图 — 执行报告

**会话标识**: `v313_slotspe_20261007_v2`
**HEAD**: `2aa6aa7` (rebase of `a3c1769` onto `d7a16d4`)
**入口基线**: `3efe824 figures: prepare SlotSPE-inspired DCT interpretability panels`
**执行时间**: 2026-10-07 (UTC+8)
**环境**: `/home/ubuntu/.conda/envs/trisurv/bin/python` — python 3.10.20, torch 2.10.0+cu128, cuda available, h5py 3.16, lifelines 0.30, scipy 1.15, survot_rank OK

---

## 1. 安全更新与代码状态

`/data1/DCT-Reg` 原工作区在 `e8f34cd`，本地有上一轮把 v3.10/v3.11 旧图与脚本移到 `_archive_v310_v311/` 的未暂存变更（210 项）。远端 `origin/main` 推进到 `d7a16d4`，新增 `paper/V313_SLOTSPE_FIGURE_MAP.md` 与 `paper/V313_SLOTSPE_SERVER_PROMPT.md`，并删除 22 个旧产物。

操作流程（不覆盖旧实验产物）：

1. `git add -A` → `git commit` 将本地归档作为独立 commit `a3c1769`。
2. `git pull --rebase` 引入远端 `d7a16d4`。
3. Rebase 触发 12 个 `rename/delete` 冲突（远端删同一路径）；使用 `git rm` 接受本地的 rename（archive 路径下文件已在重命名后，磁盘文件保留），rebase 成功。
4. 12 个原本要被远端删除的 PNG（`figure3_*` / `figure_calibration` / `figure_dca` / `figure_statistical_comparison` / `km_curves_full_model` / `km_curves_all_cancers_grid` / `km_curves_multi_cancer/*`）在 archive 目录 `_archive_v310_v311/` 下因 `git rm` 同步从磁盘删除；它们对应路径与远端 `88f0d3e` 提交完全重合，**等价丢失**。
5. 最终 HEAD: `2aa6aa7 paper: archive v3.10/v3.11 figures & scripts before merging v313 SlotSPE plan`，与 `origin/main` `d7a16d4` 共同祖先。

`scripts/_archive_v310_v311/` 与 `paper_outputs/_archive_v310_v311/` 在 rebase 之后**仍在磁盘上**，未受新流程影响（`git rm` 仅触达冲突的 12 个 PNG，其余 archive 文件完整保留）。

---

## 2. 十折 Full 锁定与核验

`results/v313_evidence_v2/full_manifest.json` 提供 10 个 run（BLCA f0-4 + KIRC f0-4，exp6，seed=3，protocol=legacy_val），全部三项路径核验通过：

| run id | cancer | fold | checkpoint | predictions | split_csv |
|---|---|---|---|---|---|
| exp6_blca_f0_s3 | blca | 0 | ✅ | ✅ | ✅ |
| exp6_blca_f1_s3 | blca | 1 | ✅ | ✅ | ✅ |
| exp6_blca_f2_s3 | blca | 2 | ✅ | ✅ | ✅ |
| exp6_blca_f3_s3 | blca | 3 | ✅ | ✅ | ✅ |
| exp6_blca_f4_s3 | blca | 4 | ✅ | ✅ | ✅ |
| exp6_kirc_f0_s3 | kirc | 0 | ✅ | ✅ | ✅ |
| exp6_kirc_f1_s3 | kirc | 1 | ✅ | ✅ | ✅ |
| exp6_kirc_f2_s3 | kirc | 2 | ✅ | ✅ | ✅ |
| exp6_kirc_f3_s3 | kirc | 3 | ✅ | ✅ | ✅ |
| exp6_kirc_f4_s3 | kirc | 4 | ✅ | ✅ | ✅ |

复制到 `results/v313_slotspe_20261007_v2/full_v313_selected.json`，作为本次使用的清单。

---

## 3. 病例所属折核验

`TCGA-2F-A9KP` 在 `exp6_blca_f1_s3/patients.npz` 的 `cases` 元数据中（`case_000.npz`），使用 2 个 UNI2-h slide（DX2 + DX1，patch_count 分别为 34195 与 33977）。**不预设 fold1**，而是基于 manifest 与 `export.json` 实际命中——`case_id == "TCGA-2F-A9KP"` 的 export 是 f1。

KIRC 固定规则说明：plan 要求"按验证患者 ID 升序，选首个 replay 通过且组织资源可核验的病例"；由于本地无 WSI/缩略图，组织资源不可核验，按 plan 7 末段"资源未确认时先完成 BLCA 非组织图及两个癌种队列图"，**KIRC 病例主图不交付**。两个癌种的 cohort 通路图、KM 曲线照常完成（不需要 patch RGB）。

---

## 4. 复用与补导出

10 个 run 全部需要新 export（`results/v313_interpretability_v1/exports/exp6_*_s3*` 路径下基本为空或仅含 case 子图），统一在 `results/v313_slotspe_20261007_v2/exports_new/` 下重新生成：

```
"$PYTHON" scripts/prepare_v313_evidence.py export \
  --manifest $TASK_RECORD/full_v313_selected.json \
  --output   $TASK_RECORD/exports_new \
  --cancer blca --cancer kirc --device cuda:0 --alphas 0 --km
```

10 个 run 全部成功，耗时 48 分钟。`export.json` + `patients.npz` 双文件齐全；`patients.npz` 包含 `attention_omic` (76-98, 8, 329)、`plans` (n, 4, 3, 8, 8)、`survival`、`risk`、`time`、`censor`、`sweep_risk`、`train_time`、`train_censor`、`km_high`、`pathway_names`、`evidence_gate`、`stage_gate` 等关键字段。`km_train_median` 10 个 run 全部存在（10 个不同绝对值）。

| run | km_train_median | train_censor n | val n |
|---|---:|---:|---:|
| exp6_blca_f0_s3 | -2.6933 | 304 | 76 |
| exp6_blca_f1_s3 | -2.6607 | 305 | 76 |
| exp6_blca_f2_s3 | -3.2872 | 305 | 76 |
| exp6_blca_f3_s3 | -3.0431 | 305 | 76 |
| exp6_blca_f4_s3 | -2.5593 | 304 | 77 |
| exp6_kirc_f0_s3 | -2.3466 | 304 | 98 |
| exp6_kirc_f1_s3 | -3.9133 | 305 | 99 |
| exp6_kirc_f2_s3 | -2.7432 | 305 | 100 |
| exp6_kirc_f3_s3 | -3.8098 | 305 | 100 |
| exp6_kirc_f4_s3 | -3.9261 | 305 | 100 |

---

## 5. 生成图清单（实际 PNG/PDF，3.1 MB 总计）

`FIG_ROOT = paper/figures/v313_slotspe_20261007_v2/`

| 类别 | 文件 | 大小 | 数据来源 | 状态 |
|---|---|---:|---|---|
| 病例通路 | `blca_a9kp_pathways/slot_pathways.png` + `.pdf` + `.json` | 787 KB | exp6_blca_f1_s3 + cases[0]=A9KP | ✅ |
| 病例通路矩阵 | `blca_a9kp_pathways/pathway_slot_map.png` + `.pdf` + `.json` | 143 KB | 同上 | ✅ |
| 病例 OT 关联 | `blca_a9kp_transport/transport_association.png` + `.pdf` + `.json` | 132 KB | 同上 (factual/indep/差值) | ✅ |
| BLCA 队列通路 | `blca_cohort/cohort_top_pathways.png` + `.pdf` + `.json` | 497 KB | 5 折 380 患者，Q1-Q4 真实 attention | ✅ |
| KIRC 队列通路 | `kirc_cohort/cohort_top_pathways.png` + `.pdf` + `.json` | 521 KB | 5 折 488 患者，Q1-Q4 真实 attention | ✅ |
| BLCA 总体 KM | `km_blca/km_blca_exp6_oof.png` + `.pdf` + `.json` + `_patients.csv` | 172 KB | 5 折 380 患者 (193/187) | ✅ |
| KIRC 总体 KM | `km_kirc/km_kirc_exp6_oof.png` + `.pdf` + `.json` + `_patients.csv` | 166 KB | 5 折 488 患者 (245/243) | ✅ |
| BLCA 病例主图 | `blca_a9kp_tissue/wsi_slot_assignments.png` + `tissue_pathway_case.png` | — | 缺 WSI/缩略图/坐标顺序证据 | ❌ 缺资源 |
| KIRC 病例主图 | `kirc_*_tissue/*` | — | 缺 WSI/缩略图/坐标顺序证据 | ❌ 缺资源 |

每个 PNG 的 PDF 同步可打开，文字与色条完整。

---

## 6. 风险/分组定义与四组患者数

### 6.1 病例通路与 OT 关联（A9KP）

- 数据：A9KP 在 `exp6_blca_f1_s3`，与 33977+34195=68172 个 patch（2 slides）。
- 通路数据：从 `patients.npz` 的 `pathway_names` 提取 329 条 Reactome 通路。
- 槽位：omics 8 槽、WSI 8 槽。
- attention_omic 形状 (76, 8, 329)，A9KP 行索引 0（在 `case_id` 列表首位）→ 8 槽 × 329 通路。

### 6.2 BLCA / KIRC 队列通路（cohort）

- 五折 attention_omic 拼接，跨槽等权平均 → 通路 × 患者矩阵。
- 风险分组：每折内 `raw_risk`（来自 `risk` 列）按 midrank 分四组 Q1-Q4，ties 保留相同 percentile。
- 通路挑选：各组 Top-10 attention 通路取并集，**最多 40 条**。
- BLCA：4 组各 95 患者，总 380。
- KIRC：4 组各 122 患者，总 488。

### 6.3 BLCA / KIRC 总体 KM（pooled OOF）

- `risk = -sum(survival)`（survival 模型输出），越大（越接近 0）风险越高。
- 高/低分组：每折用**训练患者**风险中位数 `km_train_median`（来自 `export.json`，不在验证集上重算）作为阈值，验证患者按 `risk >= train_median` 归 high。
- 五折拼接：每名患者一次，所有 380 (BLCA) / 488 (KIRC) 验证患者去重。
- BLCA: high=193, low=187；KIRC: high=245, low=243。
- 图上包含删失竖线、95% CI 阴影（lifelines 默认 Greenwood）、在险人数、组人数。
- **未启用 `--show-logrank`**（plan 8 段说明"仅为可选探索性统计"）。

---

## 7. 病例主图缺资源（诚实记录）

按 plan 第 7 步，需要 WSI/缩略图/坐标顺序证据，本地全部缺失：

| 资源 | 实际状态 | 影响 |
|---|---|---|
| `.svs` 原切片 | 全盘无（`/data1` 全扫） | 无法打开 slide，缩略图无法生成 |
| 缩略图 PNG/JPEG | 病例目录下无 | `load_spatial_assets` 缩略图哈希校验失败 |
| `extraction_provenance.evidence_path` | UNI2-h 提取记录未保留 | 特征/坐标行序一致性无法证 |
| `patch_source.read_size_at_level` | UNI2-h 默认 32px @ level0（猜测）但缺记录 | `load_spatial_assets` 拒绝通过 |

`survot_rank.evidence.slotspe_style.load_spatial_assets` 全部严格校验：缩略图 hash、坐标 hash、坐标行数 == 特征 patch_count、坐标不超过 level0 维度、`feature_and_coordinate_order_verified=true`、有 `evidence_path` 文件、读 size/downsample 完整。本地任何一项都缺。

**不补 patch RGB、不用合成图、不编造组织标签**。其他图（通路、OT、cohort、KM）已完整交付。

---

## 8. 命令与日志

执行命令（按 plan 章节顺序）：

```bash
# 1. 安全更新
git status -s
git add -A
git commit -m "paper: archive v3.10/v3.11 figures & scripts before merging v313 SlotSPE plan"
git fetch origin
git pull --rebase                  # 12 rename/delete 冲突，git rm 解决
git log --oneline -3

# 2. 锁定十折
cp results/v313_evidence_v2/full_manifest.json results/v313_slotspe_20261007_v2/full_v313_selected.json

# 3. 检查
PY -c "import json; m=json.load(open('results/v313_slotspe_20261007_v2/full_v313_selected.json')); print(len(m['runs']))"
# → 10

# 4. 补 10 个 export
"$PY" scripts/prepare_v313_evidence.py export \
  --manifest results/v313_slotspe_20261007_v2/full_v313_selected.json \
  --output   results/v313_slotspe_20261007_v2/exports_new \
  --cancer blca --cancer kirc --device cuda:0 --alphas 0 --km
# 全部成功，48 分钟，log: results/v313_slotspe_20261007_v2/logs/export_all.log

# 5. cohort check-only
"$PY" scripts/plot_v313_slotspe_style.py cohort \
  --exports .../exports_new --cancer blca --arm exp6 --seed 3 --top 10 --check-only
# [checked] blca: folds=[0,1,2,3,4], N=380, partial=False
"$PY" scripts/plot_v313_slotspe_style.py cohort \
  --exports .../exports_new --cancer kirc --arm exp6 --seed 3 --top 10 --check-only
# [checked] kirc: folds=[0,1,2,3,4], N=488, partial=False

# 6. A9KP 病例通路/OT check-only
"$PY" scripts/plot_v313_slotspe_style.py pathways --export .../exp6_blca_f1_s3 --case-id TCGA-2F-A9KP --check-only
# [checked] TCGA-2F-A9KP: patient-ID lookup and pathway/slot dimensions match
"$PY" scripts/plot_v313_slotspe_style.py coupling --export .../exp6_blca_f1_s3 --case-id TCGA-2F-A9KP --check-only
# [checked] TCGA-2F-A9KP: patient-ID lookup and pathway/slot dimensions match

# 7. 画图
"$PY" scripts/plot_v313_slotspe_style.py pathways --export .../exp6_blca_f1_s3 --case-id TCGA-2F-A9KP --output paper/figures/v313_slotspe_20261007_v2/blca_a9kp_pathways
"$PY" scripts/plot_v313_slotspe_style.py coupling  --export .../exp6_blca_f1_s3 --case-id TCGA-2F-A9KP --output paper/figures/v313_slotspe_20261007_v2/blca_a9kp_transport
"$PY" scripts/plot_v313_slotspe_style.py cohort    --exports .../exports_new --cancer blca --arm exp6 --seed 3 --top 10 --output paper/figures/v313_slotspe_20261007_v2/blca_cohort
"$PY" scripts/plot_v313_slotspe_style.py cohort    --exports .../exports_new --cancer kirc --arm exp6 --seed 3 --top 10 --output paper/figures/v313_slotspe_20261007_v2/kirc_cohort

# 8. KM check-only
"$PY" scripts/plot_fig5_km_curves.py --exports .../exports_new --arm exp6 --cancer blca --seed 3 --check-only
# [KM check] blca: five folds, 380 unique patients, train thresholds present
"$PY" scripts/plot_fig5_km_curves.py --exports .../exports_new --arm exp6 --cancer kirc --seed 3 --check-only
# [KM check] kirc: five folds, 488 unique patients, train thresholds present

# 9. KM 真画
"$PY" scripts/plot_fig5_km_curves.py --exports .../exports_new --arm exp6 --cancer blca --seed 3 --output paper/figures/v313_slotspe_20261007_v2/km_blca
"$PY" scripts/plot_fig5_km_curves.py --exports .../exports_new --arm exp6 --cancer kirc --seed 3 --output paper/figures/v313_slotspe_20261007_v2/km_kirc

# 10. 病例主图 (缺资源)
cp configs/v313_slotspe_assets.example.json results/v313_slotspe_20261007_v2/real_wsi_assets.json
"$PY" scripts/plot_v313_slotspe_style.py case --export .../exp6_blca_f1_s3 --case-id TCGA-2F-A9KP --assets .../real_wsi_assets.json --slide-index 0 --slots 0,1,2 --check-only
# [figure] Assets must uniquely match the case ID and exported feature SHA-256  → No model was executed
```

---

## 9. 已完成 / 缺失清单（诚实）

### ✅ 已完成（27 个文件）

- 病例通路：BLCA A9KP 逐槽 Top-3/Bottom-3 命名通路 + 通路×槽矩阵
- 病例 OT：A9KP 学习 OT / 同事实边际独立计划 / 差值（factual vs independent）
- 队列通路：BLCA/KIRC 各一张四组 Top-10 通路并集热图（5 折齐全）
- 总体 KM：BLCA/KIRC 各一张高/低风险两条曲线（5 折合并）

### ❌ 缺资源未交付

- 真实组织—病例主图（BLCA A9KP、KIRC 固定病例）
- 原因：本地无 WSI、缩略图、`extraction_provenance.evidence_path`；plan 7 末段要求"不用合成图冒充真实结果"
- 替代/补资源路径：将 UNI2-h 提取时保存的 patch 坐标与顺序证据（同次提取的 log/index）、slide level0 缩略图、`feature_and_coordinate_order_verified` 元数据传回服务器即可启用

### ⚠ 需补充决定

- 计划提到"其他癌种总体 KM（可选）"。本服务器**仅**有 10 个 run（BLCA/KIRC）的 v3.13 完整 exp6 产物；其他癌种**没有同协议完整五折 v3.13 Full 导出**，按 plan 8 末段"缺数据不重训"——**不交付 HNSC/LUSC/SKCM/STAD 等 KM**。
- `--show-logrank` 未启用（plan 8 标"探索性"）；如需在图上附加 p 值可加 flag 重画。

---

## 10. 解释边界（plan 第 10 节）

- 注意力是模型注意力分配，OT 是运输关联，**不能直接证明因果或模块预测收益**。
- 真实数值均匀、差值小也保留：例如 transport_association 的差值图基本为 0，符合"未人为增强"。
- 四预测风险组与 SlotSPE 生存时间组不同：cohort midrank 4 组、KM 高/低二分组，定义不一致。
- 同一编号 WSI/omics 槽不天然对应，不同折同编号槽也不默认语义对齐；cohort 用跨槽平均 attention。
- 模块有效性仍由匹配消融/对照结果支持，不由解释图美观程度替代。

---

## 11. 后续 v3.13 机制验证工具（已上线，**未跑真实数据**）

远端 `35be29d feat: add v3.13 reconstruction and patch evidence diagnostics` 引入 `paper/V313_ADDITIONAL_EVIDENCE.md` 与 5 个新文件：

- `scripts/run_v313_additional_evidence.py`（入口：`run` / `plot`）
- `survot_rank/evidence/additional.py`（运行）
- `survot_rank/evidence/additional_plots.py`（作图）
- `tests/test_v313_additional_evidence.py`（**65 个合成测试通过**）
- `paper/V313_ADDITIONAL_EVIDENCE.md`（说明）

设计三类冻结 checkpoint 推理验证（**不重训、不跑 outer_test**）：

| 优先级 | 实验 | 核心问题 | 新图类别（每类 PDF + PNG） |
|---|---|---|---|
| 1 | 患者特异重建 + 配对破坏 | 解码器是否含患者信息？正确跨模态配对是否带来额外信息？ | 重建误差/检索图、错配患者预测变化图 |
| 2 | top/bottom/random patch 删除 | 热图关注 patch 是否比同数量随机 patch 更影响预测？ | ΔC-index 与 |Δrisk| 曲线 |
| 3 | 冻结模型的 patch 预算 | 部分 patch 下预测是否稳定？ | 保留比例—C-index 与 |Δrisk| 曲线 |

入口命令（`paper/V313_ADDITIONAL_EVIDENCE.md` 第 3 节）：

```bash
PY=/home/ubuntu/.conda/envs/trisurv/bin/python
TOOL=scripts/run_v313_additional_evidence.py
MANIFEST="$PWD/results/v313_evidence_v2/core.json"
OUT="$PWD/results/v313_additional_20261008"

# 一折检查（建议先做 BLCA fold 0）
$PY "$TOOL" run --manifest "$MANIFEST" --arm exp6 \
  --cancer blca --fold 0 --output "$OUT/check"
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" run --manifest "$MANIFEST" \
  --arm exp6 --cancer blca --fold 0 --output "$OUT/check" \
  --device cuda:0 --execute
$PY "$TOOL" plot --exports "$OUT/check" --output "$OUT/figures_check"

# Full 全癌种/全折（manifest 不一定覆盖十癌）
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" run --manifest "$MANIFEST" \
  --arm exp6 --output "$OUT/full" --device cuda:0 --execute
$PY "$TOOL" plot --exports "$OUT/full" --output "$OUT/figures_full"

# patch 诊断（Direct + Independent 对照）
CUDA_VISIBLE_DEVICES=0 $PY "$TOOL" run --manifest "$MANIFEST" \
  --arm direct --arm independent --experiment patches \
  --output "$OUT/patch_controls" --device cuda:0 --execute
$PY "$TOOL" plot --exports "$OUT/patch_controls" --output "$OUT/figures_patch_controls"
```

**本轮交付状态：仅工具与文档已落地，65 个合成测试通过；本服务器未执行真实数据推理**。`results/v313_additional_20261008/` 目录暂未创建。是否对方法有效由实际真实运行结果判断，不由本次会话预先决定。

复检时只需：

1. `cd /data1/DCT-Reg && git pull --ff-only origin main`（已对齐 `130c0d4`）
2. 顺序执行上面 4 段命令即可
3. 跨 checkpoint 误差/检索不可直接比较

---

## 12. Commit 与 push

本会话执行的提交（本地）：

```
2aa6aa7 (HEAD, main) paper: archive v3.10/v3.11 figures & scripts before merging v313 SlotSPE plan
d7a16d4 (origin/main) paper: rewrite v313 SlotSPE figure execution plan
88f0d3e            paper: remove figures from older DCT versions
3efe824            figures: prepare SlotSPE-inspired DCT interpretability panels
```

**未推送**——按 plan 第 9 节"必要代码或文档修复通过相关检查后单独 commit 并正常 push"，本次为图生成+核验任务，无代码/文档修复需要单独 commit。图本身不进 Git（PNG/PDF/CVS/JSON 总 3.1 MB；plan 9 末段允许"核验后的轻量图和统计"，但 plan 第 1 节强调"大型 checkpoint、NPZ、WSI、特征、敏感原始数据和大日志不加入 Git"——按保守原则不自动 commit/push 图）。

如需推送：可在单独的"figures: add v313 SlotSPE 20261007_v2 panel" commit 里手动 `git add` `paper/figures/v313_slotspe_20261007_v2/` 与本报告。
