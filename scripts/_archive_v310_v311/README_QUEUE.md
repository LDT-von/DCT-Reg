# DCT-Reg 队列运行 cheatsheet（不跑、只看怎么跑）

本文档**只描述**如何运行 DCT-Reg 训练队列与监控，**不执行任何训练**。

## 一、两条冻结队列的口径对比

| 项 | 队列 A（uni2-h） | 队列 B（uni） |
|---|---|---|
| 脚本 | `scripts/run_dct_v310_final_cross_cancer.py` | `scripts/run_dct_v310_final_uni_queue.py` |
| `wsi_encoder` | `uni2-h` | `uni` |
| `encoding_dim` | `1536` | `1024` |
| `data_root_dir` | `/data1/TCGA-UNI2-h-features` | `/data/CPathPatchFeature` |
| `which_splits` | `5fold_uni2h` | `5fold` |
| 输出目录 | `results/dct_v3.10/robust/final/<cancer>` | `results/dct_v3.10/robust/final_uni/<cancer>` |
| 适用癌种 | uni2-h 覆盖率 100% 的：blca/hnsc/kirc/lusc/skcm/ucec | uni2-h 有缺口而 UNI 100% 的：brca/coadread/luad/stad |

**完全相同**（保持冻结 30ep 口径）：

```
max_epochs=30
bag_loss=nll_surv
dct_lambda_ipcw_rank=0.10
dct_v38_lambda_direction=0.05
dct_v38_lambda_dose=0.0
dct_v38_lambda_reconfiguration=0.0
dct_v38_warmup_epochs=0
dct_v38_ramp_epochs=0
dct_lambda_etar=0.0
dct_lambda_listwise=0.0
dct_v382_lambda_mgptr=0.0
dct_v382_adaptive_aux_weights=False
fit_bins_on_train=True
binning_mode=global_qcut
dct_slot_init_mode=deterministic
event_stratified_batches=True
event_sampling_fraction=0.0
dct_ipcw_rank_memory_size=64
dct_mix_ratio=1.0
num_patches=2048
batch_size=8
on_missing_wsi=error
```

## 二、一键 cheatsheet（你需要时复制粘贴）

### 1) 队列 A（uni2-h）—— 当前计划：kirc+ucec

```bash
# 一次性：把旧仓库的 9 个癌种 splits symlink 进当前数据集根
cd /data1/DCT-Reg
for c in brca coadread hnsc kirc luad lusc skcm stad ucec; do
  ln -sfn /data1/SurvOT-Rank/survot_rank/research/legacy/slotspe_runtime/dataset_csv/splits/5fold_uni2h/$c \
          data/dataset_csv/splits/5fold_uni2h/$c
done
ls -la data/dataset_csv/splits/5fold_uni2h/   # 应有 10 个癌种

# 0) 校验
python scripts/run_dct_v310_final_cross_cancer.py doctor --cancers kirc,ucec --gpu 0

# 1) plan（只打印，不训练）
python scripts/run_dct_v310_final_cross_cancer.py plan --cancers kirc,ucec --gpu 0

# 2) run（这才是真跑）
mkdir -p logs
nohup python -u scripts/run_dct_v310_final_cross_cancer.py run \
  --cancers kirc,ucec --folds 0,1,2,3,4 --gpu 0 --num-workers 4 \
  > logs/run_dct_v310_uni2h_queue.log 2>&1 &
echo "PID=$!"
```

### 2) 队列 B（uni）—— 当前计划：brca/coadread/luad/stad

```bash
# 0) 校验（应 OK: 4 个癌种的 UNI features/clinical/5fold splits）
python scripts/run_dct_v310_final_uni_queue.py doctor \
  --cancers brca,coadread,luad,stad --gpu 0

# 1) plan
python scripts/run_dct_v310_final_uni_queue.py plan \
  --cancers brca,coadread,luad,stad --gpu 0

# 2) run
mkdir -p logs
nohup python -u scripts/run_dct_v310_final_uni_queue.py run \
  --cancers brca,coadread,luad,stad --folds 0,1,2,3,4 --gpu 0 --num-workers 4 \
  > logs/run_dct_v310_uni_queue.log 2>&1 &
echo "PID=$!"
```

### 3) 监控（任意终端，可重复执行）

```bash
# 一次性快照（无副作用）
python scripts/monitor_unified.py

# 持续刷新（每 5 秒）
python scripts/monitor_unified.py --watch 5
```

监控覆盖：GPU 显存/利用率、队列锁（哪些 fold 正在跑）、已完成 fold、日志 tail。

### 4) 替代：用包装脚本打命令

```bash
./scripts/launch_and_monitor.sh uni2h "kirc,ucec" 0 5
./scripts/launch_and_monitor.sh uni    "brca,coadread,luad,stad" 0 5
```

它只打印要跑的 4 行命令（doctor / plan / run / monitor），**不会自动起任何东西**。

## 三、3.11 怎么迁移

如果将来要做 DCT v3.11，**不要**碰队列脚本里的口径；只动：

1. **新建脚本** `scripts/run_dct_v311_final_uni2h_queue.py` 与 `run_dct_v311_final_uni_queue.py`（复制当前两个文件即可）。
2. 改两个常量：
   - 配置文件：`"configs/dct_v311_xxx.yaml"`
   - `survot_method`：v3.11 自己的 method id
   - 新的 λ（`dct_v311_lambda_*`）和辅助开关
3. 输出目录：`results/dct_v3.11/robust/final/<cancer>`（与 3.10 隔离）。
4. 锁目录共用 `results/.locks/`，但加 `_v311` 后缀避免冲突：
   - 改 `_queue_runtime.scheduler_lock_path`（或本地复写一份），输出 `.<kind>_gpu_<gpu>_v311.lock`。

队列脚本本身（`Job` / `run_queue` / doctor）**不需要重写**，参数化即复用。

## 四、当前仓库里没做的事

- 没在 GPU 上跑过任何 fold（doctor 校验显示 kirc/ucec 的 5fold_uni2h 分区还没链进数据集根；UNI 队列已经齐备）。
- 队列 B 脚本只做了静态语法校验（`py_compile`），没有真起 doctor/run。
- 监控脚本同样只静态校验。

**等你下一步指令再决定要不要触发 doctor / plan / run。**
