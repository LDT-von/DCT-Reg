#!/usr/bin/env python3
"""E047: Plot BLCA efficiency comparison figures.

Generates two scatter plots:
  - Left:  BLCA C-index vs Peak GPU Memory (MiB)
  - Right: BLCA C-index vs Median Forward Latency (ms)

Data sources:
  - DCT v3.13 Full: measured C-index + measured costs (includes OT reconstruction)
  - External methods: published C-index from literature + measured core inference costs
    (labeled as "public score + measured cost" to distinguish from same-condition comparison)

Usage:
    python plot_e047_efficiency.py [--input INPUT.json] [--output OUTPUT_DIR]
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path(__file__).parent / "e047_efficiency_data.json"
DEFAULT_OUTPUT = Path(__file__).parent / "figures"


def load_data(path: Path) -> dict:
    """Load and validate E047 efficiency data."""
    data = json.loads(path.read_text(encoding="utf-8"))
    
    required_dct = ("label", "cindex", "memory_mib", "latency_ms")
    for key in required_dct:
        if key not in data.get("dct", {}):
            raise ValueError(f"DCT data missing required field: {key}")
    
    for i, ext in enumerate(data.get("external", [])):
        for key in ("label", "cindex", "memory_mib", "latency_ms"):
            if key not in ext:
                raise ValueError(f"External method {i} missing: {key}")
    
    return data


def plot_e047_tradeoff(data: dict, output_dir: Path) -> dict:
    """Generate the E047 efficiency tradeoff figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    points = []
    
    # DCT point
    dct = data["dct"]
    points.append({
        "id": "dct",
        "label": dct["label"],
        "ours": True,
        "cindex": dct["cindex"],
        "memory_mib": dct["memory_mib"],
        "latency_ms": dct["latency_ms"],
    })
    
    # External methods
    for ext in data.get("external", []):
        points.append({
            "id": ext.get("id", ext["label"].lower().replace("-", "_")),
            "label": ext["label"],
            "ours": False,
            "cindex": ext["cindex"],
            "memory_mib": ext["memory_mib"],
            "latency_ms": ext["latency_ms"],
            "note": ext.get("note", ""),
        })
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("BLCA Efficiency Comparison: DCT v3.13 vs External Methods", 
                 fontweight="bold", fontsize=14, y=0.98)
    
    dct_color = "#e69f00"
    ext_colors = plt.get_cmap("tab10")(np.arange(len(points) - 1) % 10)
    
    # Plot left: Memory vs C-index
    ax = axes[0]
    for i, p in enumerate(points):
        color = dct_color if p["ours"] else ext_colors[i - 1]
        size = 250 if p["ours"] else 100
        marker = "*" if p["ours"] else "o"
        
        ax.scatter(p["memory_mib"], p["cindex"], 
                   s=size, marker=marker, color=color,
                   edgecolors="black" if p["ours"] else "none",
                   linewidths=1.5 if p["ours"] else 0,
                   zorder=3)
        
        offset_y = 8 if i % 2 == 0 else -15
        ax.annotate(p["label"], 
                   (p["memory_mib"], p["cindex"]),
                   xytext=(8, offset_y),
                   textcoords="offset points",
                   fontsize=9, fontweight="bold" if p["ours"] else "normal")
        
        if p.get("note"):
            ax.annotate(p["note"],
                       (p["memory_mib"], p["cindex"]),
                       xytext=(8, -20 if i % 2 == 0 else 8),
                       textcoords="offset points",
                       fontsize=7, style="italic",
                       color="#666666")
    
    ax.set_xlabel("Peak GPU Allocated Memory (MiB)", fontsize=11)
    ax.set_ylabel("BLCA C-index", fontsize=11)
    ax.set_title("Performance vs Memory", fontsize=12, fontweight="bold")
    ax.grid(ls="--", alpha=0.35)
    ax.margins(0.12)
    ax.set_xlim(left=0)
    
    # Plot right: Latency vs C-index
    ax = axes[1]
    
    # Check if DCT latency is much larger
    dct_latency = points[0]["latency_ms"]
    max_ext_latency = max(p["latency_ms"] for p in points[1:])
    
    # Use log scale for latency if DCT is >10x larger than max external
    use_log_latency = dct_latency > max_ext_latency * 10
    
    for i, p in enumerate(points):
        color = dct_color if p["ours"] else ext_colors[i - 1]
        size = 250 if p["ours"] else 100
        marker = "*" if p["ours"] else "o"
        
        if use_log_latency:
            # Use log scale, place DCT at right edge
            x_val = p["latency_ms"]
        else:
            x_val = p["latency_ms"]
        
        ax.scatter(x_val, p["cindex"],
                   s=size, marker=marker, color=color,
                   edgecolors="black" if p["ours"] else "none",
                   linewidths=1.5 if p["ours"] else 0,
                   zorder=3)
        
        offset_y = 8 if i % 2 == 0 else -15
        ax.annotate(p["label"],
                   (x_val, p["cindex"]),
                   xytext=(8, offset_y),
                   textcoords="offset points",
                   fontsize=9, fontweight="bold" if p["ours"] else "normal")
        
        if p.get("note"):
            ax.annotate(p["note"],
                       (x_val, p["cindex"]),
                       xytext=(8, -20 if i % 2 == 0 else 8),
                       textcoords="offset points",
                       fontsize=7, style="italic",
                       color="#666666")
    
    ax.set_xlabel("Median Forward Wall-clock Time (ms)" + (" (log scale)" if use_log_latency else ""), 
                  fontsize=11)
    ax.set_ylabel("BLCA C-index", fontsize=11)
    ax.set_title("Performance vs Latency", fontsize=12, fontweight="bold")
    ax.grid(ls="--", alpha=0.35)
    ax.margins(0.12)
    ax.set_xlim(left=0)
    
    if use_log_latency:
        ax.set_xscale("log")
        # Add annotation for log scale
        ax.annotate("DCT latency\n(OT reconstruction):\n~493 ms", 
                   (dct_latency, points[0]["cindex"]),
                   xytext=(-80, 30),
                   textcoords="offset points",
                   fontsize=8, style="italic",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"))
    
    # Legend
    handles = [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor=dct_color,
                   markersize=18, label="DCT v3.13 Full (ours)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=12, label="External methods (literature)"),
    ]
    axes[1].legend(handles=handles, fontsize=9, loc="lower right")
    
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    
    # Footnote
    footnote = (
        "DCT v3.13: measured on RTX 5090, batch=1, 2048 patches, 30 repeats, 5 warmup. "
        "Latency includes OT reconstruction (~493 ms).\n"
        "External methods: measured with reference-like architectures, random init weights. "
        "External C-index: published values from original papers (not same-condition results). "
        "Memory: peak allocated GPU memory (MiB)."
    )
    fig.text(0.5, 0.02, footnote, ha="center", fontsize=8, style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.8))
    
    # Save
    files = {}
    for ext in ("png", "pdf", "svg"):
        path = output_dir / f"e047_efficiency_tradeoff.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        files[ext] = str(path)
    plt.close(fig)
    
    # Save data summary
    summary = {
        "dct": data["dct"],
        "external": data["external"],
        "plot_settings": {
            "log_latency_scale": use_log_latency
        }
    }
    (output_dir / "e047_efficiency_data.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Input JSON with efficiency data")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output directory for figures")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[E047] Input file not found: {args.input}")
        return 1
    
    print(f"[E047] Loading data from {args.input}")
    data = load_data(args.input)
    
    print(f"[E047] Generating figures...")
    files = plot_e047_tradeoff(data, args.output)
    
    for ext, path in files.items():
        print(f"  {ext}: {path}")
    
    print(f"[E047] Done. Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
