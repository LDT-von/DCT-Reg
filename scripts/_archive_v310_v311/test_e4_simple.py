#!/usr/bin/env python3
"""Minimal E4 test to verify model loading and intervention logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from argparse import Namespace

from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
    SurvivalDatasetFactory, SurvivalDataset
)
from survot_rank.training.model_factory import get_model


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Checkpoint path
    checkpoint_path = (
        "results/dct_v3.10_experiments/robust/direction_only/blca/blca/"
        "SurvOTRank_dct_transport_intervention_consistency/"
        "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep/"
        "model_best_s0.pth"
    )
    
    print(f"Testing with checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    
    # Load config
    config_path = Path(__file__).parent.parent / 'configs' / 'dct_v310_directional_regularized_transport.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create args (matching Direction Only variant)
    args = Namespace(
        study='blca',
        survot_method='dct_transport_intervention_consistency',  # Ablation parent
        
        # Data paths
        data_root_dir='/data1/TCGA-UNI2-h-features',
        data_path='/data1/DCT-Reg/data/dataset_csv',
        wsi_encoder='uni2-h',
        on_missing_wsi='error',
        
        # Data config
        rna_format='Pathways',
        signature='combine',
        label_col='survival_months_dss',
        n_classes=4,
        n_bins=4,
        num_patches=4096,
        encoding_dim=1536,
        num_genes=None,
        num_workers=4,
        
        # Training
        batch_size=8,
        max_epochs=50,
        lr=5e-4,
        bag_loss='nll_surv',
        alpha_surv=0.0,
        
        # Slot config
        slot_num_wsi=8,
        slot_num_omics=8,
        slot_iters=3,
        temperature=1.0,
        topk_ratio=0.25,
        top_k_method='random',
        
        # Model config
        otehv2_eps=0.01,
        otehv2_iter=30,
        otehv2_heads=8,
        otehv2_layers=2,
        otehv2_dropout=0.1,
        dct_num_stages=4,
        
        # DCT objectives (Direction Only: no IPCW, only direction)
        dct_lambda_ipcw_rank=0.0,  # Disabled
        dct_ipcw_rank_margin=0.5,
        dct_ipcw_rank_temperature=0.1,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=512,
        dct_v38_lambda_direction=0.05,  # Enabled
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        
        # DCT parameters
        dct_anchor_momentum=0.95,
        dct_evidence_cost_weight=0.1,
        dct_evidence_mass_floor=0.01,
        dct_evidence_marginal_strength=1.0,
        dct_geometry_reliability_strength=1.0,
        dct_coupling_projection_iters=10,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.1,
        dct_mix_ratio=0.5,
        
        dct_v38_direction_margin=0.2,
        dct_v38_dose_margin=0.3,
        dct_v38_reconfiguration_margin=0.1,
        dct_v38_temperature=0.1,
        dct_v38_alpha_mid=0.5,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=5,
        dct_v38_ramp_epochs=10,
        dct_v38_dose_every=5,
        
        dct_lambda_etar=0.0,
        dct_lambda_listwise=0.0,
        dct_v382_lambda_mgptr=0.0,
        fet_lambda_sparse=0.0,
        fet_lambda_faith=0.0,
        spt_prog_cost=0.20,  # Changed from 'learned' to float
        rg_eps_start=0.1,
        rg_eps_anneal=0.99,
        
        # Legacy model parameters (will be set after dataset factory)
        omic_sizes=None,
        wsi_projection_dim=256,
        wsi_dim=256,
        rna_dim=256,
        gene_dim=256,
        rank_weight=8.0,
        graph_reg_weight=8.0,
        dropout=0.25,
    )
    
    print("\n1. Creating dataset factory...")
    dataset_factory = SurvivalDatasetFactory(
        study='blca',
        data_path=args.data_path,
        rna_format='Pathways',
        signature='combine',
        n_bins=4,
        label_col='survival_months_dss',
        num_genes=None,
        num_patches=4096,
        clinical_feature_cols=None,
        binning_mode='global_qcut',
        which_splits='5fold_uni2h',
    )
    
    # Update args with omic sizes
    args.omic_sizes = dataset_factory.omic_sizes
    omic_input_dim = sum(dataset_factory.omic_sizes)
    omic_names = dataset_factory.omic_names
    
    print(f"Omic input dim: {omic_input_dim}")
    print(f"Omic names: {omic_names}")
    print(f"Omic sizes: {dataset_factory.omic_sizes}")
    
    print("\n2. Creating model...")
    model = get_model(
        method=args.survot_method,
        args=args,
        omic_input_dim=omic_input_dim,
        omic_names=omic_names,
        pathway_names=None
    )
    
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model type: {type(model)}")
    
    print("\n3. Loading train dataset to configure reference...")
    wsi_path_train = Path('/data1/TCGA-UNI2-h-features') / 'blca' / 'uni2-h' / 'pt_files'
    train_dataset = SurvivalDataset(
        dataset_factory=dataset_factory,
        wsi_path=str(wsi_path_train),
        split_key='train',
        fold=0,
        encoding_dim=1536,
        on_missing_wsi='error',
    )
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Extract train labels
    train_times = train_dataset.label_df[dataset_factory.label_col].values
    train_censor = train_dataset.label_df[dataset_factory.censorship_var].values
    
    print(f"Train times shape: {train_times.shape}")
    print(f"Train censorship shape: {train_censor.shape}")
    
    # Configure model with train reference
    if hasattr(model, 'configure_train_reference'):
        model.configure_train_reference(train_times, train_censor)
        print("Configured train reference")
    
    print("\n4. Loading checkpoint...")
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    print(f"Checkpoint keys (first 10): {list(state_dict.keys())[:10]}")
    
    # Load state dict
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")
    
    model = model.to(device)
    model.eval()
    
    print("\n5. Checking model attributes...")
    print(f"Has risk_anchor_costs: {hasattr(model, 'risk_anchor_costs')}")
    if hasattr(model, 'risk_anchor_costs'):
        print(f"  Shape: {model.risk_anchor_costs.shape}")
        print(f"  Type: {type(model.risk_anchor_costs)}")
    
    print(f"Has event_reference: {hasattr(model, 'event_reference')}")
    if hasattr(model, 'event_reference'):
        print(f"  Shape: {model.event_reference.shape}")
    
    # Try to extract anchors
    print("\n6. Extracting prognostic anchors...")
    try:
        if hasattr(model, 'risk_anchor_costs'):
            anchors = model.risk_anchor_costs
            print(f"Found risk_anchor_costs with shape: {anchors.shape}")
            # Shape is [num_stages, 2, num_slots_wsi, slot_dim_wsi, slot_dim_omics]
            # We need [2, embedding_dim] for low/high risk
            # Extract first stage, flatten the slots
            if anchors.dim() == 5:
                # [num_stages, 2, ...]
                low_risk = anchors[0, 0].flatten()  # First stage, low risk
                high_risk = anchors[-1, 1].flatten()  # Last stage, high risk
                print(f"Extracted anchors:")
                print(f"  Low risk shape: {low_risk.shape}")
                print(f"  High risk shape: {high_risk.shape}")
                print(f"  Distance: {torch.norm(high_risk - low_risk).item():.4f}")
        elif hasattr(model, 'event_reference'):
            ref = model.event_reference
            print(f"Found event_reference with shape: {ref.shape}")
            low_risk = ref[0]
            high_risk = ref[-1]
            print(f"Extracted anchors:")
            print(f"  Low risk shape: {low_risk.shape}")
            print(f"  High risk shape: {high_risk.shape}")
            print(f"  Distance: {torch.norm(high_risk - low_risk).item():.4f}")
        else:
            print("ERROR: Cannot find prognostic anchors!")
            print(f"Available attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
    except Exception as e:
        print(f"ERROR extracting anchors: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n7. Loading test dataset...")
    wsi_path = Path('/data1/TCGA-UNI2-h-features') / 'blca' / 'uni2-h' / 'pt_files'
    test_dataset = SurvivalDataset(
        dataset_factory=dataset_factory,
        wsi_path=str(wsi_path),
        split_key='val',
        fold=0,
        encoding_dim=1536,
        on_missing_wsi='error',
    )
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\n8. Testing forward pass...")
    sample = test_dataset[0]
    print(f"Sample type: {type(sample)}")
    print(f"Sample length: {len(sample)}")
    
    # Try forward pass
    try:
        with torch.no_grad():
            if isinstance(sample, dict):
                wsi = sample['wsi'].unsqueeze(0).to(device) if 'wsi' in sample else sample['x_path'].unsqueeze(0).to(device)
                omics = sample['omics'].unsqueeze(0).to(device) if 'omics' in sample else None
            else:
                # Tuple format: (x_path, x_omic, ...)
                x_path = sample[0]
                x_omic = sample[1] if len(sample) > 1 else None
                
                # Handle list of patches
                if isinstance(x_path, list):
                    x_path = torch.stack(x_path)  # Stack list of patches
                if not isinstance(x_path, torch.Tensor):
                    x_path = torch.tensor(x_path)
                wsi = x_path.unsqueeze(0).to(device)
                
                # Prepare omics as pathway format
                if x_omic is not None and isinstance(x_omic, list):
                    # Multiple pathways - create x_omic1, x_omic2, ...
                    omic_kwargs = {}
                    for i, pathway in enumerate(x_omic):
                        if isinstance(pathway, list):
                            pathway = torch.cat([t.flatten() if t.dim() > 1 else t for t in pathway])
                        if not isinstance(pathway, torch.Tensor):
                            pathway = torch.tensor(pathway)
                        omic_kwargs[f'x_omic{i+1}'] = pathway.unsqueeze(0).to(device)
                else:
                    omic_kwargs = {}
            
            print(f"WSI shape: {wsi.shape}")
            if omic_kwargs:
                print(f"Number of pathways: {len(omic_kwargs)}")
            
            # Forward
            output = model(x_wsi=wsi, **omic_kwargs)
            
            print(f"Output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Output tuple length: {len(output)}")
                print(f"  First element shape: {output[0].shape if hasattr(output[0], 'shape') else type(output[0])}")
            elif hasattr(output, 'shape'):
                print(f"Output shape: {output.shape}")
            
            print("\nSUCCESS: Model forward pass works!")
    except Exception as e:
        print(f"ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
