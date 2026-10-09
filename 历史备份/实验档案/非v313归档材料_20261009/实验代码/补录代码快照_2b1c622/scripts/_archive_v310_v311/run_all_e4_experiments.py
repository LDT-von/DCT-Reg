#!/usr/bin/env python3
"""One-click runner for all E4 Continuous Intervention Audit experiments.

This script automatically runs E4 experiments for:
- 3 variants: direction_only, ipcw_only, full
- 5 folds: 0, 1, 2, 3, 4
- BLCA dataset

Total: 15 experiments

Usage:
    python scripts/run_all_e4_experiments.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
PYTHON = "/home/ubuntu/.conda/envs/trisurv/bin/python"
SCRIPT = "scripts/e4_continuous_intervention_audit_v2.py"
CONFIG = "configs/dct_v310_directional_regularized_transport.yaml"
STUDY = "blca"
DEVICE = "cuda:0"
ALPHAS = "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"

# Variants to test
VARIANTS = {
    "direction_only": {
        "checkpoint_base": "results/dct_v3.10_experiments/robust/direction_only/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_direction_only_blca_50ep",
        "desc": "Direction Only (λ_dir=0.05, λ_ipcw=0)"
    },
    "ipcw_only": {
        "checkpoint_base": "results/dct_v3.10_experiments/robust/ipcw_only/blca/blca/SurvOTRank_dct_transport_intervention_consistency/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_ipcw_only_blca_50ep",
        "desc": "IPCW Only (λ_dir=0, λ_ipcw=0.10)"
    },
    "full": {
        "checkpoint_base": "results/dct_v3.10_experiments/robust/full/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_full_blca_50ep",
        "desc": "Full Model (λ_dir=0.05, λ_ipcw=0.10)"
    }
}

FOLDS = [0, 1, 2, 3, 4]

OUTPUT_BASE = "results/e4_intervention_audit"


def check_checkpoint_exists(checkpoint_path):
    """Check if checkpoint file exists."""
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
        return False
    return True


def run_e4_experiment(variant, fold, checkpoint_path, output_path, dry_run=False):
    """Run a single E4 experiment."""
    cmd = [
        PYTHON,
        SCRIPT,
        "--checkpoint", checkpoint_path,
        "--config", CONFIG,
        "--study", STUDY,
        "--fold", str(fold),
        "--output", output_path,
        "--alphas", ALPHAS,
        "--device", DEVICE
    ]
    
    print(f"\nRunning: {variant} fold {fold}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_path}")
    
    if dry_run:
        print(f"  [DRY RUN] Command: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed with error:")
        print(e.stderr)
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all E4 experiments")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--variants", default="direction_only,ipcw_only,full",
        help="Comma-separated list of variants to run"
    )
    parser.add_argument(
        "--folds", default="0,1,2,3,4",
        help="Comma-separated list of folds to run"
    )
    args = parser.parse_args()
    
    # Parse variants and folds
    selected_variants = args.variants.split(',')
    selected_folds = [int(f) for f in args.folds.split(',')]
    
    # Create output directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Print header
    print("=" * 80)
    print("E4 CONTINUOUS INTERVENTION AUDIT - ONE-CLICK RUNNER")
    print("=" * 80)
    print(f"Study: {STUDY}")
    print(f"Variants: {', '.join(selected_variants)}")
    print(f"Folds: {', '.join(map(str, selected_folds))}")
    print(f"Total experiments: {len(selected_variants) * len(selected_folds)}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"Device: {DEVICE}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 80)
    
    # Check all checkpoints exist
    print("\n📋 Checking checkpoints...")
    missing_checkpoints = []
    
    for variant in selected_variants:
        if variant not in VARIANTS:
            print(f"⚠️  Unknown variant: {variant}")
            continue
        
        variant_info = VARIANTS[variant]
        print(f"\n{variant}: {variant_info['desc']}")
        
        for fold in selected_folds:
            checkpoint_path = os.path.join(
                variant_info['checkpoint_base'],
                f"model_best_s{fold}.pth"
            )
            
            if check_checkpoint_exists(checkpoint_path):
                print(f"  ✓ Fold {fold}: {checkpoint_path}")
            else:
                missing_checkpoints.append((variant, fold, checkpoint_path))
    
    if missing_checkpoints:
        print("\n❌ Missing checkpoints:")
        for variant, fold, path in missing_checkpoints:
            print(f"  {variant} fold {fold}: {path}")
        
        if not args.dry_run:
            response = input("\nContinue anyway? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                return
    
    # Run experiments
    print("\n" + "=" * 80)
    print("🚀 RUNNING E4 EXPERIMENTS")
    print("=" * 80)
    
    results = []
    start_time = datetime.now()
    
    for i, variant in enumerate(selected_variants):
        if variant not in VARIANTS:
            continue
        
        variant_info = VARIANTS[variant]
        print(f"\n{'='*80}")
        print(f"VARIANT {i+1}/{len(selected_variants)}: {variant}")
        print(f"Description: {variant_info['desc']}")
        print(f"{'='*80}")
        
        for fold in selected_folds:
            checkpoint_path = os.path.join(
                variant_info['checkpoint_base'],
                f"model_best_s{fold}.pth"
            )
            
            output_path = os.path.join(
                OUTPUT_BASE,
                f"{variant}_{STUDY}_fold{fold}.csv"
            )
            
            # Skip if checkpoint doesn't exist
            if not os.path.exists(checkpoint_path):
                print(f"\n⏭️  Skipping {variant} fold {fold} (checkpoint missing)")
                results.append((variant, fold, "SKIPPED"))
                continue
            
            # Run experiment
            success = run_e4_experiment(
                variant, fold, checkpoint_path, output_path, dry_run=args.dry_run
            )
            
            results.append((variant, fold, "SUCCESS" if success else "FAILED"))
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for _, _, status in results if status == "SUCCESS")
    failed_count = sum(1 for _, _, status in results if status == "FAILED")
    skipped_count = sum(1 for _, _, status in results if status == "SKIPPED")
    
    print(f"Total experiments: {len(results)}")
    print(f"  ✓ Success: {success_count}")
    print(f"  ✗ Failed: {failed_count}")
    print(f"  ⏭ Skipped: {skipped_count}")
    print(f"Duration: {duration}")
    
    print("\nDetailed results:")
    for variant, fold, status in results:
        emoji = "✓" if status == "SUCCESS" else "✗" if status == "FAILED" else "⏭"
        print(f"  {emoji} {variant:20s} fold {fold}: {status}")
    
    # List output files
    if success_count > 0:
        print(f"\n📁 Output files in {OUTPUT_BASE}:")
        output_files = sorted(Path(OUTPUT_BASE).glob("*.csv"))
        for f in output_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  {f.name} ({size_mb:.2f} MB)")
    
    print("\n" + "=" * 80)
    print("🎉 ALL E4 EXPERIMENTS COMPLETED!")
    print("=" * 80)
    
    if not args.dry_run:
        print(f"\nNext steps:")
        print(f"1. Analyze results: python scripts/analyze_e4_results.py")
        print(f"2. Visualize: python scripts/visualize_e4_results.py")
        print(f"3. Generate report: python scripts/generate_e4_report.py")


if __name__ == '__main__':
    main()
