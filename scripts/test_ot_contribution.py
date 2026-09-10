#!/usr/bin/env python3
"""
对抗性删除 OT 实验

通过 monkey-patch forward 方法，将 OT 输出替换为均匀分布，
对比 Full（有 OT）和 No-OT（无 OT）的 C-index 差异。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from survot_rank.cli import add_project_paths
add_project_paths()

from sksurv.metrics import concordance_index_censored


def inject_no_ot_mode(model):
    """为模型注入 no-OT 模式：替换 transport plans 为均匀分布。

    实现方式：调用完整 forward 获得 logits，然后 monkey-patch
    ``_plans_from_cost_tensor`` 让它返回均匀 plan。这样所有下游逻辑
    （event_tokens, event_gate, hazard, ...）和原模型完全一致，只是
    plan 是均匀的，OT 的几何信息被完全擦除。
    """

    original_plans_from_cost = model._plans_from_cost_tensor

    def uniform_plans_from_cost(costs, rows, cols, epoch, *args, **kwargs):
        """Return uniform transport plans matching the original shape."""
        B = costs.size(0)
        # 从 args / checkpoint 形状推断 slot 数
        Kw = costs.size(-2)
        Ko = costs.size(-1)
        n_stages = costs.size(1)  # [B, n_stages, n_geo, Kw, Ko]
        n_geo = costs.size(2)
        uniform_plan = (
            torch.ones(B, Kw, Ko, device=costs.device, dtype=costs.dtype) / (Kw * Ko)
        )
        # 构造 plans 列表：[n_stages] 个 list，每个 list 含 [n_geo] 个 [B,Kw,Ko]
        plans = [
            [uniform_plan for _ in range(n_geo)]
            for _ in range(n_stages)
        ]
        # 第二个返回值（info dict）保持原 API
        return plans, {"uniform_injected": True}

    model._plans_from_cost_tensor = uniform_plans_from_cost
    model._no_ot_mode = True
    return model


def evaluate_model(model, val_loader, device, parsed):
    """评估模型 C-index

    Risk 计算公式（与 core_utils._calculate_risk 一致）：
        survival = cumprod(1 - sigmoid(hazards), dim=1)
        risk = -sum(survival, dim=1)
    这是离散时间生存分析的标准做法，不能用 -logits.mean()。
    """
    try:
        from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import _process_data_and_forward
    except ImportError:
        _process_data_and_forward = None

    model.eval()
    all_risk_scores = []
    all_event_times = []
    all_events = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            if _process_data_and_forward is not None:
                out, y_disc, event_time, censorship = _process_data_and_forward(
                    parsed, model, data, device, test=True
                )
                logits = out[0] if isinstance(out, tuple) else out
                # 正确的 risk 计算：cumprod(1 - sigmoid(hazards))
                hazards = logits
                survival = torch.cumprod(1 - torch.sigmoid(hazards), dim=1)
                risk = -torch.sum(survival, dim=1).cpu().numpy()
                times = event_time.cpu().numpy()
                c = censorship.cpu().numpy()  # c=0 is event, c=1 is censored
                events = (c < 0.5).astype(bool)
            else:
                # Fallback: manual processing
                batch_x_wsi = data[0].to(device)  # WSI
                batch_x_omic = [omic.to(device) if isinstance(omic, torch.Tensor) else omic for omic in data[1]]  # Omics
                outputs = model(x_wsi=batch_x_wsi, x_omics=batch_x_omic)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                hazards = logits
                survival = torch.cumprod(1 - torch.sigmoid(hazards), dim=1)
                risk = -torch.sum(survival, dim=1).cpu().numpy()
                times = data[3].cpu().numpy()  # event_time
                c = data[4].cpu().numpy()  # censorship: c=0 is event, c=1 is censored
                events = (c < 0.5).astype(bool)

            all_risk_scores.extend(risk.tolist())
            all_event_times.extend(times.tolist())
            all_events.extend(events.tolist())

            # Debug
            if batch_idx == 0:
                print(f"    Debug batch 0: events type={type(events)}, dtype={events.dtype}, first={events[0] if len(events) > 0 else None}")
                print(f"    Debug batch 0: risk range = [{risk.min():.3f}, {risk.max():.3f}], mean = {risk.mean():.3f}")
                print(f"    Debug batch 0: hazards shape = {hazards.shape}")

    if len(all_risk_scores) == 0:
        return None, 0

    # 转换为 numpy 数组
    risk_scores = np.array(all_risk_scores)
    y_time_arr = np.array(all_event_times, dtype=np.float64)
    y_event_arr = np.array(all_events, dtype=bool)  # True = event (uncensored)

    # 诊断：risk 分布
    n_unique = len(set(risk_scores.tolist()))
    print(f"    [DIAG] n={len(risk_scores)}, risk unique={n_unique}/{len(risk_scores)}")
    print(f"    [DIAG] risk range=[{risk_scores.min():.3f}, {risk_scores.max():.3f}], std={risk_scores.std():.4f}")

    # 计算 C-index
    # sksurv 接口: concordance_index_censored(event_indicator, event_time, estimate)
    # 注意: 第一个参数是 event_indicator，不是 event_time
    c_index, _, _, _, _ = concordance_index_censored(
        y_event_arr, y_time_arr, risk_scores
    )

    return c_index, len(all_risk_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--study", type=str, default="blca")
    parser.add_argument("--rna_format", type=str, default="Pathways")
    parser.add_argument("--signature", type=str, default="combine")
    parser.add_argument("--method", type=str, default="dct_v310_directional_regularized_transport")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"OT 贡献对抗性测试")
    print(f"{'='*60}")
    
    # 加载模型和数据
    print(f"\n[1] 加载 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # 创建配置
    from argparse import Namespace
    from survot_rank.training.model_factory import get_model
    from survot_rank.config import load_config
    
    # 尝试从 experiment_settings.txt 加载
    exp_dir = Path(args.checkpoint).parent.parent.parent
    settings_file = exp_dir / "experiment_settings.txt"
    
    if settings_file.exists():
        import json as json_mod
        settings = eval(open(settings_file).read())
        parsed = Namespace(**settings)
        # 确保有所有必需的 data_path
        if not hasattr(parsed, 'data_path') or parsed.data_path is None:
            parsed.data_path = '/data1/dataset_csv'
        if not hasattr(parsed, 'data_root_dir'):
            # slotspe features at /data1/DCT-Reg/data/slotspe_pt_features/{study}/pt_files
            parsed.data_root_dir = '/data1/DCT-Reg/data/slotspe_pt_features'
        if not hasattr(parsed, 'split_dir'):
            parsed.split_dir = f"{parsed.data_path}/splits/5fold/{parsed.study}"
        if not hasattr(parsed, 'encoding_dim'):
            parsed.encoding_dim = 1536
        if not hasattr(parsed, 'wsi_projection_dim'):
            parsed.wsi_projection_dim = 256
        if not hasattr(parsed, 'slot_num_wsi'):
            parsed.slot_num_wsi = 8
        if not hasattr(parsed, 'slot_num_omics'):
            parsed.slot_num_omics = 8
        if not hasattr(parsed, 'slot_iters'):
            parsed.slot_iters = 3
        if not hasattr(parsed, 'batch_size'):
            parsed.batch_size = 8
        if not hasattr(parsed, 'num_workers'):
            parsed.num_workers = 4
        # 使用 slotspe 作为 wsi_encoder, study 作为编码器子目录
        parsed.wsi_encoder = parsed.study  # blca
        parsed.on_missing_wsi = 'zero'  # 允许缺失 WSI 时零填充
        parsed.n_classes = 4  # 离散时间生存 4 bins
        parsed.otehv2_layers = 3  # checkpoint 训练时用的层数
        parsed.k_start = args.fold
        parsed.k_end = args.fold + 1
    else:
        # 使用默认配置
        parsed = Namespace()
        parsed.study = args.study
        parsed.rna_format = args.rna_format
        parsed.signature = args.signature
        parsed.n_classes = 4
        parsed.label_col = 'survival_months_dss'
        parsed.num_genes = None
        parsed.num_patches = 2048
        parsed.clinical_feature_cols = None
        parsed.binning_mode = 'global_qcut'
        parsed.data_path = '/data1/dataset_csv'
        parsed.data_root_dir = '/data1/DCT-Reg/data/slotspe_pt_features'
        parsed.split_dir = f"{parsed.data_path}/splits/5fold/{parsed.study}"
        parsed.encoding_dim = 1536
        parsed.wsi_projection_dim = 256
        parsed.slot_num_wsi = 8
        parsed.slot_num_omics = 8
        parsed.slot_iters = 3
        parsed.batch_size = 8
        parsed.num_workers = 4
        parsed.wsi_encoder = parsed.study
        parsed.on_missing_wsi = 'zero'
    
    parsed.folder = args.fold
    parsed.survot_method = args.method

    # ⚠️ 关键：checkpoint 训练时用的是 otehv2_layers=3
    # 不设这个会让 TransformerEncoder 多出一层，权重加载失败 (12 missing keys)
    parsed.otehv2_layers = 3

    # 加载数据
    print("[2] 加载数据...")
    try:
        from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
            SurvivalDatasetFactory,
        )
        from survot_rank.training.train_runner import get_split

        factory = SurvivalDatasetFactory(
            study=parsed.study,
            data_path=getattr(parsed, 'data_path', None),
            rna_format=parsed.rna_format,
            signature=parsed.signature,
            n_bins=parsed.n_classes,
            label_col=parsed.label_col,
            num_genes=parsed.num_genes,
            num_patches=parsed.num_patches,
            clinical_feature_cols=(
                [c.strip() for c in parsed.clinical_feature_cols.split(",") if c.strip()]
                if getattr(parsed, "clinical_feature_cols", None)
                else None
            ),
            binning_mode=getattr(parsed, "binning_mode", "global_qcut"),
        )

        if parsed.rna_format in ("Pathways", "RNASeq", "GeneEmbedding"):
            rna_cases = set(factory.gene_data_df.columns)
            factory.clinical_df = factory.clinical_df[
                factory.clinical_df["case id"].isin(rna_cases)
            ].reset_index(drop=True)

        train_data, val_data, _, val_loader = get_split(parsed, factory, args.fold)
        parsed.omic_sizes = factory.omic_sizes
        parsed.omic_names = factory.omic_names
        parsed.pathway_names = getattr(factory, "pathway_names", None)

        # ⚠️ 关键：checkpoint 训练时用的是 otehv2_layers=2
        # 不设这个会让 TransformerEncoder 多出一层，权重加载失败 (12 missing keys)
        parsed.otehv2_layers = 2

        if parsed.rna_format == "RNASeq":
            omics_input_dim = (
                factory.num_genes if factory.num_genes is not None else factory.omic_sizes
            )
        elif parsed.rna_format == "GeneEmbedding":
            omics_input_dim = 768
        else:
            omics_input_dim = None
        
        print(f"    数据: {parsed.study} fold {args.fold}, n_val={len(val_data)}")
        
    except Exception as e:
        print(f"    数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建模型
    print("[3] 创建模型...")
    try:
        model = get_model(
            method=parsed.survot_method,
            args=parsed,
            omic_input_dim=omics_input_dim,
            omic_names=parsed.omic_names,
            pathway_names=parsed.pathway_names,
        )

        # 配置训练参考
        model.configure_train_reference(
            train_data.label_df[factory.label_col].to_numpy(),
            train_data.label_df[factory.censorship_var].to_numpy(),
        )

        # 重新注册 censor buffers 为 checkpoint 的大小，避免 size mismatch
        # (split 不同时 censor buffer 长度会变)
        ckpt_censor_times = checkpoint.get('dct_censor_times', None)
        ckpt_censor_surv  = checkpoint.get('dct_censor_survival', None)
        if ckpt_censor_times is not None and ckpt_censor_surv is not None:
            new_n = int(ckpt_censor_times.shape[0])
            # 用 dummy 值替换，重新注册 buffer，再 load_state_dict
            import torch.nn as nn
            # 把旧 buffer 替换
            for nm in ('dct_censor_times', 'dct_censor_survival'):
                if nm in model.state_dict():
                    old = getattr(model, nm, None)
                    if old is not None and not isinstance(old, nn.Parameter):
                        # 删除旧的，重新注册
                        delattr(model, nm)
            model.register_buffer('dct_censor_times', torch.zeros(new_n))
            model.register_buffer('dct_censor_survival', torch.zeros(new_n))

        # 加载权重
        model_state_keys = set(model.state_dict().keys())
        filtered_state_dict = {}
        skipped = []

        for key, value in checkpoint.items():
            if key in model_state_keys:
                model_shape = model.state_dict()[key].shape
                checkpoint_shape = value.shape
                if model_shape == checkpoint_shape:
                    filtered_state_dict[key] = value
                else:
                    skipped.append(f"{key}: {checkpoint_shape} -> {model_shape}")
            else:
                filtered_state_dict[key] = value

        if skipped:
            print(f"    ⚠️  Skipped {len(skipped)} size-mismatched parameters")
            for s in skipped[:3]:
                print(f"        {s}")

        missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
        if missing:
            print(f"    ⚠️  Missing keys: {len(missing)} -> {missing[:3]}")
        if unexpected:
            print(f"    ⚠️  Unexpected keys: {len(unexpected)} -> {unexpected[:3]}")

        # 重新配置 train_reference（buffer 大小改变后必须重新写入）
        model.configure_train_reference(
            train_data.label_df[factory.label_col].to_numpy(),
            train_data.label_df[factory.censorship_var].to_numpy(),
        )

        print(f"    模型: {parsed.survot_method}")

    except Exception as e:
        print(f"    模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    parsed.cur_epoch = 999  # 用于推理模式
    
    # 评估 Full 模型
    print(f"\n[4] 评估 Full 模型 (有 OT)...")
    cindex_full, n_full = evaluate_model(model, val_loader, device, parsed)
    print(f"    Full C-index: {cindex_full:.4f} (n={n_full})")
    
    # 注入 no-OT 模式
    print(f"\n[5] 注入 No-OT 模式...")
    model = inject_no_ot_mode(model)
    print("    No-OT 模式已启用")
    
    # 评估 No-OT 模型
    print(f"\n[6] 评估 No-OT 模型 (无 OT)...")
    cindex_no_ot, n_no_ot = evaluate_model(model, val_loader, device, parsed)
    print(f"    No-OT C-index: {cindex_no_ot:.4f} (n={n_no_ot})")
    
    # 计算 OT 贡献
    print(f"\n{'='*60}")
    print(f"结果汇总")
    print(f"{'='*60}")
    
    if cindex_full is not None and cindex_no_ot is not None:
        ot_contribution = cindex_full - cindex_no_ot
        ot_percentage = (ot_contribution / cindex_full) * 100 if cindex_full > 0 else 0
        
        print(f"\n| 配置 | C-index |")
        print(f"|------|---------|")
        print(f"| Full (有 OT) | {cindex_full:.4f} |")
        print(f"| No-OT (无 OT) | {cindex_no_ot:.4f} |")
        print(f"| **OT 贡献** | **{ot_contribution:+.4f} ({ot_percentage:+.1f}%)** |")
        
        print(f"\n判定标准:")
        if ot_contribution > 0.03:
            print("  ✅ OT 对预测有 **实质贡献** (> 0.03)")
            verdict = "substantial"
        elif ot_contribution > 0.01:
            print("  ⚠️  OT 对预测有 **中等贡献** (0.01 ~ 0.03)")
            verdict = "moderate"
        elif ot_contribution > -0.01:
            print("  ⚠️  OT 对预测 **贡献很小** (-0.01 ~ 0.01)")
            verdict = "minimal"
        else:
            print("  ❌ No-OT 反而 **更好** (< -0.01)，说明 OT 可能有干扰")
            verdict = "negative"
        
        result = {
            "fold": args.fold,
            "checkpoint": str(args.checkpoint),
            "study": parsed.study,
            "method": parsed.survot_method,
            "full_cindex": float(cindex_full),
            "no_ot_cindex": float(cindex_no_ot),
            "ot_contribution": float(ot_contribution),
            "ot_percentage": float(ot_percentage),
            "n_full": n_full,
            "n_no_ot": n_no_ot,
            "verdict": verdict
        }
    else:
        result = {
            "fold": args.fold,
            "full_cindex": float(cindex_full) if cindex_full else None,
            "no_ot_cindex": float(cindex_no_ot) if cindex_no_ot else None,
            "error": "Could not compute C-index"
        }
        verdict = "error"
    
    # 保存结果
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存: {args.output}")
    
    return result


if __name__ == "__main__":
    main()
