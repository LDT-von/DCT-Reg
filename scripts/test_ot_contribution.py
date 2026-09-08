#!/usr/bin/env python3
"""
对抗性删除 OT 实验 v2

更简单的实现：直接修改 forward 方法，将 OT 输出置零

假设: 如果 OT 对预测有贡献，去掉 OT 后 C-index 应该显著下降
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from argparse import Namespace

import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

# 添加项目路径
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from survot_rank.research.methods.catalog import METHOD_CATALOG


def inject_no_ot_mode(model):
    """
    为模型注入 no-OT 模式
    通过 monkey-patch forward 方法实现
    """
    # 创建 no-OT forward
    def forward_no_ot(self, **kwargs):
        """Forward without OT transport plans - use uniform transport"""
        x_wsi_proj = self.wsi_mlp(kwargs["x_wsi"])
        x_omics = self._encode_omics(kwargs)
        slots_wsi = self.slot_attention_wsi(x_wsi_proj)
        slots_omic = self.slot_attention_omic(x_omics)

        # 关键修改：用均匀分布替换 OT plans
        B = slots_wsi.size(0)
        T = self.num_classes
        Kw = slots_wsi.size(1)
        Ko = slots_omic.size(1)
        
        # 均匀运输计划（无结构信息）
        uniform_plan = (
            torch.ones(B, Kw, Ko, device=slots_wsi.device, dtype=slots_wsi.dtype) 
            / (Kw * Ko)
        )
        
        # 创建 dummy plans（4个 stage）
        n_stages = getattr(self.args, 'dct_num_stages', 4)
        dummy_plans = [
            uniform_plan.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, Kw, Ko]
            for _ in range(n_stages)
        ]
        
        # 用 dummy plans 计算 event tokens
        event_tokens = self._stagewise_events(slots_wsi, slots_omic, dummy_plans)
        event_tokens = event_tokens + self.stage_embedding.unsqueeze(0)
        event_tokens = self.event_norm(self.event_encoder(event_tokens))

        event_logits = self.event_hazard(event_tokens)
        gate = torch.softmax(self.event_gate(event_tokens).squeeze(-1), dim=1)
        logits = torch.einsum("be,bec->bc", gate, event_logits)

        # 返回 logits，aux_loss = 0（不用于评估）
        return logits, torch.tensor(0.0, device=logits.device)
    
    # Monkey-patch
    model.forward = forward_no_ot.__get__(model, type(model))
    model._no_ot_mode = True
    
    return model


def evaluate_model(model, data, batch_size=32):
    """评估模型 C-index"""
    model.eval()
    all_risk_scores = []
    all_event_times = []
    all_events = []
    
    features = data["features"]
    y_times = data["y_time"]
    y_events = data["y_event"]
    n_samples = len(y_times)
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            
            # 准备 batch 数据
            batch_x_wsi = torch.tensor(features["x_wsi"][i:batch_end], dtype=torch.float32)
            batch_x_omic_list = []
            for j in range(len(features["x_omics"])):
                batch_x_omic_list.append(torch.tensor(features["x_omics"][j][i:batch_end], dtype=torch.float32))
            
            batch_x_omic = batch_x_omic_list
            
            # Forward
            try:
                outputs = model(x_wsi=batch_x_wsi, x_omics=batch_x_omic)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                
                # 风险分数 = -mean(logits)
                risk = -logits.mean(dim=-1).cpu().numpy()
                
                all_risk_scores.extend(risk.tolist())
                all_event_times.extend(y_times[i:batch_end])
                all_events.extend([bool(e) for e in y_events[i:batch_end]])
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                continue
    
    if len(all_risk_scores) == 0:
        return None, 0
    
    # 计算 C-index
    y_time_arr = np.array(all_event_times)
    y_event_arr = np.array(all_events, dtype=bool)
    y_censored_arr = ~y_event_arr  # True = censored
    risk_scores = np.array(all_risk_scores)
    
    c_index, _, _, _, _ = concordance_index_censored(
        y_time_arr, y_censored_arr, risk_scores
    )
    
    return c_index, len(all_risk_scores)


def load_test_data(checkpoint_path, fold):
    """加载测试数据"""
    # 从 checkpoint 路径推断数据路径
    base_dir = Path(checkpoint_path).parent.parent.parent.parent.parent.parent
    data_dir = base_dir / "data"
    
    # 查找数据文件
    cancer = "blca"  # 从路径中提取
    data_file = data_dir / f"{cancer}_fold{fold}_test.pkl"
    
    if not data_file.exists():
        # 尝试其他路径
        data_file = Path("/data1/DCT-Reg/data") / f"{cancer}_fold{fold}_test.pkl"
    
    if not data_file.exists():
        print(f"Data file not found. Tried: {data_file}")
        return None
    
    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"OT 贡献对抗性测试")
    print(f"{'='*60}")
    
    # 加载 checkpoint
    print(f"\n[1] 加载 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # 加载模型
    print("[2] 创建模型...")
    method_key = checkpoint.get("method", "dct_v310_directional_regularized_transport")
    
    # 创建模型实例
    spec = METHOD_CATALOG.get(method_key)
    if spec is None:
        spec = METHOD_CATALOG["dct_v310_directional_regularized_transport"]
    
    # 加载配置
    from survot_rank.config import load_config
    from survot_rank.training.model_factory import get_model
    config_path = checkpoint.get("config_path", "configs/dct_v310_directional_regularized_transport.yaml")
    try:
        config_dict = load_config(config_path)
        # 转换为 namespace
        model_args = Namespace(**config_dict)
    except Exception as e:
        print(f"Config load failed: {e}")
        return
    
    model_args.fold = args.fold
    model_args.gpu = -1
    
    # 设置必需的参数
    model_args.omic_sizes = getattr(model_args, 'omic_sizes', [173, 80, 42, 189, 80])
    model_args.n_classes = getattr(model_args, 'n_classes', 4)
    model_args.encoding_dim = getattr(model_args, 'encoding_dim', 1536)
    model_args.wsi_projection_dim = getattr(model_args, 'wsi_projection_dim', 256)
    model_args.slot_num_wsi = getattr(model_args, 'slot_num_wsi', 8)
    model_args.slot_num_omics = getattr(model_args, 'slot_num_omics', 8)
    model_args.slot_iters = getattr(model_args, 'slot_iters', 3)
    model_args.rna_format = getattr(model_args, 'rna_format', 'Pathways')
    model_args.bagging_strategy = getattr(model_args, 'bagging_strategy', 'standard')
    
    # 设置 omic 相关参数
    model_args.omic_names = getattr(model_args, 'omic_names', None)
    model_args.pathway_names = getattr(model_args, 'pathway_names', None)
    
    # 创建模型
    try:
        # 尝试从数据中获取 omic_input_dim
        omic_input_dim = getattr(model_args, 'omic_input_dim', [173, 80, 42, 189, 80])
        
        model = get_model(
            method=method_key,
            args=model_args,
            omic_input_dim=omic_input_dim,
            omic_names=model_args.omic_names,
            pathway_names=model_args.pathway_names
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"    模型: {spec.display_name}")
    except Exception as e:
        print(f"    模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载数据
    print("[3] 加载测试数据...")
    data = load_test_data(args.checkpoint, args.fold)
    
    if data is None:
        print("无法加载测试数据")
        return
    
    # 评估 Full 模型
    print(f"\n[4] 评估 Full 模型 (有 OT)...")
    cindex_full, n_full = evaluate_model(model, data)
    print(f"    Full C-index: {cindex_full:.4f} (n={n_full})")
    
    # 注入 no-OT 模式
    print(f"\n[5] 注入 No-OT 模式...")
    model = inject_no_ot_mode(model)
    print("    No-OT 模式已启用")
    
    # 评估 No-OT 模型
    print(f"\n[6] 评估 No-OT 模型 (无 OT)...")
    cindex_no_ot, n_no_ot = evaluate_model(model, data)
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
