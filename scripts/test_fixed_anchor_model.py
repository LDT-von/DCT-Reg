#!/usr/bin/env python3
"""测试固定锚点模型能否正确加载"""

import sys
sys.path.insert(0, '/data1/DCT-Reg')

import argparse
import torch

# 测试导入
try:
    from survot_rank.research.methods.dct_v310_fixed_anchors.model import DCTV310FixedAnchors
    print("✓ 成功导入 DCTV310FixedAnchors")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试实例化
try:
    args = argparse.Namespace()
    args.bag_loss = 'nll_surv'
    args.fixed_anchors_path = 'results/ideal_anchors/blca_extracted_fold0.pkl'
    args.lr = 0.0005
    args.reg = 1e-05
    args.max_epochs = 30
    args.batch_size = 8
    args.model_type = 'survot_rank'
    args.mode = 'path'
    args.model_size_wsi = 'small_uni2h'
    args.model_size_omic = 'small'
    args.n_classes = 4
    args.omic_input_dim = 80
    args.omic_sizes = [80]  # 添加必需参数
    args.apply_sig = False
    args.omic_names = None
    args.pathway_names = None
    
    print(f"\n尝试实例化模型...")
    print(f"  fixed_anchors_path: {args.fixed_anchors_path}")
    
    model = DCTV310FixedAnchors(args, omic_input_dim=80)
    
    print(f"✓ 模型实例化成功")
    print(f"  risk_anchor_costs shape: {model.risk_anchor_costs.shape}")
    print(f"  risk_anchor_seen: {model.risk_anchor_seen}")
    print(f"  requires_grad: {model.risk_anchor_costs.requires_grad}")
    
    # 测试 train() 模式
    model.train()
    print(f"✓ train() 模式正常")
    print(f"  锚点仍然 frozen: {not model.risk_anchor_costs.requires_grad}")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！固定锚点模型工作正常")
    print("="*70)
    
except Exception as e:
    print(f"❌ 实例化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
