#!/usr/bin/env python3
"""快速测试读取器消融实验是否工作"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# 测试检查点
checkpoint = "results/backups/direction_only_frozen_bug_20260903_173509/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep/model_best_s0.pth"

print("加载检查点...")
ckpt = torch.load(checkpoint, map_location='cpu')

print(f"检查点键: {ckpt.keys() if isinstance(ckpt, dict) else 'state_dict only'}")
print(f"检查点类型: {type(ckpt)}")

if isinstance(ckpt, dict):
    if 'args' in ckpt:
        print(f"发现 args: {type(ckpt['args'])}")
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
else:
    state = ckpt

print(f"\n模型参数数量: {len(state)}")
print(f"前 10 个键:")
for i, k in enumerate(list(state.keys())[:10]):
    print(f"  {k}: {state[k].shape}")

# 查找融合模块
fusion_keys = [k for k in state.keys() if 'fusion' in k]
print(f"\n融合模块参数 ({len(fusion_keys)} 个):")
for k in fusion_keys[:15]:
    print(f"  {k}: {state[k].shape}")

print("\n✓ 检查点加载成功")
