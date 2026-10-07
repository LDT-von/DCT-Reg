#!/usr/bin/env python3
"""Auditably prepare UNI2-h HDF5 features for unmodified official SlotSPE.

Default mode is read-only: it validates slide coverage, HDF5 schema, and the
estimated PyTorch output size.  Passing --write creates one <slide>.pt tensor
per input in a separate directory and writes a SHA-256 manifest.  Existing
outputs are never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import torch


DEFAULT_DATA_ROOT = Path("/data1/DCT-Reg/third_party/SlotSPE/dataset_csv")
DEFAULT_FEATURE_ROOT = Path("/data1/TCGA-UNI2-h-features")
EXPECTED_DIMENSION = 1536


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="blca", help="TCGA study name, e.g. blca")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--features-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data1/DCT-Reg/data/slotspe_pt_features"),
        help="Separate destination root; <study>/pt_files is appended.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Convert validated HDF5 features and write a SHA-256 manifest.",
    )
    return parser.parse_args()


def split_slide_names(entries: Iterable[object]) -> list[str]:
    slides: set[str] = set()
    for entry in entries:
        if pd.isna(entry):
            continue
        for slide in str(entry).split(", "):
            slides.add(Path(slide.strip()).stem)
    return sorted(slides)


def load_expected_slides(data_root: Path, study: str) -> list[str]:
    clinical_path = data_root / "clinical" / "all" / f"{study}.csv"
    clinical = pd.read_csv(clinical_path)
    if "wsi" not in clinical:
        raise ValueError(f"{clinical_path} has no 'wsi' column")
    return split_slide_names(clinical["wsi"])


def inspect_feature(path: Path) -> tuple[tuple[int, int], int]:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise ValueError(f"{path}: missing 'features' dataset")
        dataset = handle["features"]
        shape = tuple(dataset.shape)
        dtype = np.dtype(dataset.dtype)
    if len(shape) != 3 or shape[0] != 1 or shape[2] != EXPECTED_DIMENSION:
        raise ValueError(f"{path}: expected (1, patches, {EXPECTED_DIMENSION}), found {shape}")
    if dtype != np.dtype("float32"):
        raise ValueError(f"{path}: expected float32, found {dtype}")
    return (shape[1], shape[2]), int(np.prod(shape[1:]) * dtype.itemsize)


def destination_root(output_dir: Path, study: str) -> Path:
    return output_dir / study.lower() / "pt_files"


def convert_one(source: Path, destination: Path) -> dict[str, object]:
    with h5py.File(source, "r") as handle:
        features = np.asarray(handle["features"][:], dtype=np.float32)
    tensor = torch.from_numpy(np.squeeze(features, axis=0).copy())
    if tensor.ndim != 2 or tensor.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(f"{source}: unexpected converted shape {tuple(tensor.shape)}")
    torch.save(tensor, destination)
    return {
        "slide": source.stem,
        "input_h5": str(source),
        "output_pt": str(destination),
        "input_sha256": sha256_file(source),
        "output_sha256": sha256_file(destination),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def main() -> None:
    args = parse_args()
    study = args.study.lower()
    slides = load_expected_slides(args.data_root, study)
    source_dir = args.features_root / study / "uni2-h" / "pt_files"
    target_dir = destination_root(args.output_dir, study)
    missing = [slide for slide in slides if not (source_dir / f"{slide}.h5").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing HDF5 features; first: {missing[:5]}")

    estimated_bytes = 0
    patch_counts: list[int] = []
    for slide in slides:
        shape, size = inspect_feature(source_dir / f"{slide}.h5")
        patch_counts.append(shape[0])
        estimated_bytes += size

    report = {
        "study": study,
        "slides": len(slides),
        "source_dir": str(source_dir),
        "target_dir": str(target_dir),
        "feature_dimension": EXPECTED_DIMENSION,
        "min_patches": min(patch_counts),
        "max_patches": max(patch_counts),
        "estimated_tensor_gib": round(estimated_bytes / 2**30, 3),
        "write_requested": args.write,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.write:
        print("Read-only preflight passed. Re-run with --write to create .pt features.")
        return

    if target_dir.exists() and any(target_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = target_dir.parent / "manifest.csv"
    with manifest_path.open("x", newline="", encoding="utf-8") as handle:
        fields = ["slide", "input_h5", "output_pt", "input_sha256", "output_sha256", "shape", "dtype"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, slide in enumerate(slides, start=1):
            record = convert_one(source_dir / f"{slide}.h5", target_dir / f"{slide}.pt")
            record["shape"] = json.dumps(record["shape"])
            writer.writerow(record)
            handle.flush()
            print(f"[{index}/{len(slides)}] {slide}")
    print(f"Wrote {len(slides)} tensors and manifest: {manifest_path}")


if __name__ == "__main__":
    main()
