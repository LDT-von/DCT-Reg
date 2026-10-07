#!/usr/bin/env python3
"""
Quick experiment to validate transport improvements on BLCA.
Tests the revolutionary transport fix with multi-resolution anchors,
temporal contrast loss, and curriculum learning.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = "configs/dct_v310_transport_fix.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--python-bin", default=sys.executable, help="Python binary")
    parser.add_argument("--dry-run", action="store_true", help="Print command only")
    args = parser.parse_args()
    
    os.chdir(REPO_ROOT)
    
    command = [
        args.python_bin,
        "-m",
        "survot_rank.cli",
        "train",
        "--config",
        CONFIG,
        "--gpu",
        str(args.gpu),
    ]
    
    print("="*80)
    print("TRANSPORT FIX EXPERIMENT - BLCA")
    print("="*80)
    print(f"Config: {CONFIG}")
    print(f"GPU: {args.gpu}")
    print(f"Command: {' '.join(command)}")
    print("="*80)
    
    if args.dry_run:
        print("[DRY RUN] Command prepared but not executed")
        return 0
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("PYTHONUNBUFFERED", "1")
    
    print("\nStarting training...")
    result = subprocess.run(command, env=env)
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
