"""Data preparation for SlotSPE-style DCT figures; no model execution."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .manifest import sha256, split_ids


def load_case(export_dir, case_id):
    """Resolve a selected case through export.json, never its numeric suffix."""
    root = Path(export_dir).resolve()
    meta = json.loads((root / "export.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 1:
        raise ValueError("Require an audited schema-1 export")
    matches = [c for c in meta.get("cases", []) if c["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one recorded case {case_id}")
    record = matches[0]
    case_path = (root / record["file"]).resolve()
    if case_path.parent != root:
        raise ValueError("Case file must be inside its recorded export directory")
    with np.load(case_path, allow_pickle=False) as data:
        case = {key: np.asarray(data[key]) for key in data.files}
    with np.load(root / "patients.npz", allow_pickle=False) as data:
        ids = data["case_ids"].astype(str)
        where = np.flatnonzero(ids == case_id)
        if len(where) != 1:
            raise ValueError("Case ID does not occur once in patients.npz")
        i = int(where[0])
        names = data["pathway_names"].astype(str)
        if not np.isclose(float(case["risk"][0]), float(data["risk"][i]), rtol=1e-5, atol=2e-5):
            raise ValueError("Selected-case risk disagrees with the patient-ID lookup")
        time, censor = float(data["time"][i]), int(data["censor"][i])
    aw, ao = case["attention_wsi"][0], case["attention_omic"][0]
    _attention(aw)
    _attention(ao)
    if len(names) != ao.shape[1] or len(set(names)) != len(names):
        raise ValueError("Pathway names must uniquely match the attention columns")
    if not np.isclose(-case["survival"][0].sum(), case["risk"][0], atol=2e-5):
        raise ValueError("Expected DCT risk = -sum(survival), higher means worse")
    return dict(root=root, meta=meta, record=record, arrays=case, case_id=case_id,
                names=names, time=time, censor=censor,
                source=dict(export_sha256=sha256(root / "export.json"),
                            case_sha256=sha256(case_path),
                            patients_sha256=sha256(root / "patients.npz")))


def _attention(value):
    if value.ndim != 2 or not np.isfinite(value).all() or (value < 0).any():
        raise ValueError("Expected finite nonnegative slot-by-token attention")
    if not np.allclose(value.sum(axis=1), 1, atol=2e-4):
        raise ValueError("Expected final pooling/prototype-rollout attention with row sum 1")


def transport_maps(plans, gate, attention_wsi, attention_omic):
    """Match v3.13 geometry-average, column-normalization, then stage fusion."""
    plans, gate = np.asarray(plans), np.asarray(gate)
    aw, ao = np.asarray(attention_wsi), np.asarray(attention_omic)
    _attention(aw)
    _attention(ao)
    if plans.ndim != 4 or plans.shape[0] != len(gate):
        raise ValueError("Plans must be [stage, geometry, WSI slot, omics slot]")
    if plans.shape[-2:] != (aw.shape[0], ao.shape[0]):
        raise ValueError("OT slot axes disagree with the two attention maps")
    if not np.isfinite(plans).all() or (plans < 0).any():
        raise ValueError("Invalid transport plan")
    if not np.isfinite(gate).all() or (gate < 0).any() or gate.sum() <= 0:
        raise ValueError("Invalid stage gate")
    q = gate / gate.sum()
    consensus = plans.mean(axis=1)
    normalized = consensus / np.maximum(consensus.sum(axis=1, keepdims=True), 1e-8)
    coupling = np.einsum("s,swo->wo", q, normalized)
    # Keep distinct WSI/omics indices; equal slot counts must not hide axis bugs.
    omics_spatial = np.einsum("wo,wn->on", coupling, aw)
    slot_pathway = coupling @ ao
    patch_pathway = aw.T @ slot_pathway
    mass = plans.sum(axis=(-2, -1), keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Transport plan has zero total mass")
    independent = plans.sum(axis=-1, keepdims=True) * plans.sum(axis=-2, keepdims=True) / mass
    raw = np.einsum("s,swo->wo", q, consensus)
    product = np.einsum("s,swo->wo", q, independent.mean(axis=1))
    return dict(omics_spatial=omics_spatial, slot_pathway=slot_pathway,
                patch_pathway=patch_pathway, normalized_coupling=coupling,
                transport=raw, independent=product, residual=raw - product)


def top_patch_rows(weights, slide_index, patch_index, *, slide, count=5):
    """Rank actual sampled rows; exclude padding and duplicate original patches."""
    weights = np.asarray(weights)
    si, pi = np.asarray(slide_index), np.asarray(patch_index)
    if weights.ndim != 1 or weights.shape != si.shape or weights.shape != pi.shape:
        raise ValueError("Patch scores and recorded sampling indices disagree")
    if not np.isfinite(weights).all():
        raise ValueError("Nonfinite patch score")
    keep, seen = [], set()
    for i in np.argsort(-weights, kind="stable"):
        if si[i] == slide and pi[i] >= 0 and int(pi[i]) not in seen:
            keep.append(int(i)); seen.add(int(pi[i]))
            if len(keep) == count:
                break
    return np.asarray(keep, dtype=int)


def within_fold_percentile(risk):
    """Midranks with equal scores tied; no mixing of cross-fold risk scales."""
    risk = np.asarray(risk, dtype=float).reshape(-1)
    if not len(risk) or not np.isfinite(risk).all():
        raise ValueError("Invalid patient risk")
    _, inverse, counts = np.unique(risk, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    return (cumulative[inverse] - counts[inverse] / 2) / len(risk)


def collect_attention(export_root, *, cancer, arm="exp6", seed=3, allow_partial=False):
    """Pool one audited validation partition using slot-mean attention only."""
    records, seen, universe, canonical_names = [], set(), None, None
    for path in sorted(Path(export_root).resolve().rglob("export.json")):
        if any(".bak" in p or p.startswith("backup") for p in path.parts):
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        run = meta.get("run", {})
        actual_arm = "exp6" if run.get("arm") == "full" else run.get("arm")
        if (run.get("cancer"), actual_arm, run.get("seed")) != (cancer, arm, seed):
            continue
        if meta.get("schema_version") != 1 or run.get("protocol") != "legacy_val":
            raise ValueError("Require schema-1 legacy_val development exports")
        split_path = Path(run["split_csv"])
        if not split_path.is_absolute():
            split_path = path.parent / split_path
        if sha256(split_path) != meta.get("hashes", {}).get("split_csv"):
            raise ValueError(f"Split hash differs: {path}")
        split = split_ids(split_path)
        current = set(split["train"]) | set(split["val"])
        if universe is None:
            universe = current
        elif current != universe:
            raise ValueError("Different patient universes across folds")
        with np.load(path.with_name("patients.npz"), allow_pickle=False) as data:
            if "attention_omic" not in data.files:
                raise ValueError(f"Re-export attention_omic for {run['id']}")
            ids = data["case_ids"].astype(str)
            risk, ao = np.asarray(data["risk"]).reshape(-1), np.asarray(data["attention_omic"])
            names = data["pathway_names"].astype(str)
        if len(set(ids)) != len(ids) or set(ids) != set(split["val"]) or seen & set(ids):
            raise ValueError("Incomplete, duplicate, or overlapping validation patients")
        if ao.ndim != 3 or ao.shape[0] != len(ids) or ao.shape[2] != len(names) or len(risk) != len(ids):
            raise ValueError("Patient/pathway attention axes disagree")
        if not np.isfinite(ao).all() or (ao < 0).any() or not np.allclose(ao.sum(-1), 1, atol=2e-4):
            raise ValueError("Invalid real attention_omic")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate pathway names")
        if canonical_names is None:
            canonical_names = names
        elif set(names) != set(canonical_names):
            raise ValueError("Pathway definitions differ across folds")
        reorder = [list(names).index(n) for n in canonical_names]
        pct = within_fold_percentile(risk)
        groups = np.minimum((pct * 4).astype(int), 3)
        records.append(dict(fold=int(run["fold"]), fold_ids=np.full(len(ids), int(run["fold"])), ids=ids, risk=risk, percentile=pct,
                            groups=groups, attention=ao.mean(axis=1)[:, reorder],
                            source=dict(run_id=run["id"], export=str(path),
                                        export_sha256=sha256(path),
                                        patients_sha256=sha256(path.with_name("patients.npz")))))
        seen.update(ids)
    folds = sorted(r["fold"] for r in records)
    if not records or len(set(folds)) != len(folds) or any(f not in range(5) for f in folds):
        raise ValueError("No matching exports, duplicate folds, or invalid fold numbers")
    if not allow_partial and (folds != list(range(5)) or seen != universe):
        raise ValueError(f"Require all five folds and the complete cohort; found folds {folds}, N={len(seen)}")
    records.sort(key=lambda r: r["fold"])
    combined = {k: np.concatenate([r[k] for r in records], axis=0)
                for k in ("fold_ids", "ids", "risk", "percentile", "groups", "attention")}
    combined.update(names=canonical_names, folds=folds, sources=[r["source"] for r in records],
                    cancer=cancer, arm=arm, seed=seed, partial=seen != universe or len(folds) != 5)
    return combined


def group_pathways(cohort, top=10):
    """Four group means and union of their top pathways; preserve raw weights."""
    if top < 1:
        raise ValueError("Top pathway count must be positive")
    means, counts = [], []
    for g in range(4):
        mask = cohort["groups"] == g
        counts.append(int(mask.sum()))
        if not mask.any():
            raise ValueError(f"No patients in Q{g+1}; do not manufacture a group profile")
        means.append(cohort["attention"][mask].mean(axis=0))
    means = np.stack(means, axis=1)  # pathways x four groups
    selected = set()
    for g in range(4):
        selected.update(np.argsort(-means[:, g], kind="stable")[:top].tolist())
    selected = sorted(selected, key=lambda i: (-float(means[i].max()), str(cohort["names"][i])))
    values = means[selected]
    return dict(indices=selected, names=cohort["names"][selected], mean=values,
                delta=values - values.mean(axis=1, keepdims=True), counts=counts)


def load_spatial_assets(case, manifest_path, slide=0):
    """Read actual thumbnail/coordinates with a declared extraction-order proof."""
    from PIL import Image
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = case["record"]["features"]
    if not 0 <= slide < len(features):
        raise ValueError("Slide index does not belong to this recorded case")
    feature = features[slide]
    matches = [a for a in manifest.get("features", [])
               if a.get("case_id") == case["case_id"] and a.get("feature_sha256") == feature["sha256"]]
    if len(matches) != 1:
        raise ValueError("Assets must uniquely match the case ID and exported feature SHA-256")
    asset = matches[0]
    def resolve(value):
        if not value:
            raise ValueError("An actual asset path is missing")
        p = Path(value)
        return p if p.is_absolute() else manifest_path.parent / p
    prov = asset.get("extraction_provenance", {})
    if not prov.get("feature_and_coordinate_order_verified") or not resolve(prov.get("evidence_path")).is_file():
        raise ValueError("Need the original feature/coordinate row-order verification record")
    system = asset.get("coordinate_system", {})
    if (system.get("level"), system.get("anchor"), system.get("axes"), system.get("origin")) != (0, "upper_left", ["x", "y"], [0, 0]):
        raise ValueError("Require explicit level-0 upper-left x/y coordinates")
    coords_path = resolve(asset.get("coords_path"))
    if sha256(coords_path) != asset.get("coords_sha256"):
        raise ValueError("Coordinate hash differs")
    if coords_path.suffix.lower() == ".npy":
        coords = np.load(coords_path, allow_pickle=False)
    elif coords_path.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with h5py.File(coords_path) as h:
            coords = h[asset.get("coords_key", "coords")][...]
    elif coords_path.suffix.lower() == ".csv":
        with coords_path.open(encoding="utf-8-sig", newline="") as f:
            coords = np.array([[float(r["x"]), float(r["y"])] for r in csv.DictReader(f)])
    else:
        raise ValueError("Coordinates must be NPY, HDF5, or x/y CSV")
    if coords.shape != (int(feature["patch_count"]), 2) or not np.isfinite(coords).all() or (coords < 0).any():
        raise ValueError("Coordinate rows must match all original feature rows")
    thumb = asset.get("thumbnail", {})
    thumb_path = resolve(thumb.get("path"))
    if sha256(thumb_path) != thumb.get("sha256"):
        raise ValueError("Thumbnail hash differs")
    thumbnail = Image.open(thumb_path).convert("RGB")
    if list(thumbnail.size) != thumb.get("dimensions"):
        raise ValueError("Thumbnail dimensions differ from extraction metadata")
    dimensions = np.asarray(asset.get("slide", {}).get("level0_dimensions"), dtype=float)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or (dimensions <= 0).any():
        raise ValueError("Need verified slide level-0 width/height")
    if (coords >= dimensions).any():
        raise ValueError("Coordinates exceed slide bounds")
    source = asset.get("patch_source", {})
    size, downsample = np.asarray(source.get("read_size_at_level"), dtype=float), source.get("level_downsample")
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any() or not downsample or float(downsample) <= 0:
        raise ValueError("Need actual extraction read size and level downsample")
    return dict(asset=asset, resolve=resolve, coords=coords, thumbnail=thumbnail,
                scale=np.asarray(thumbnail.size) / dimensions,
                footprint=size * float(downsample), source=source,
                coords_sha256=sha256(coords_path), thumbnail_sha256=sha256(thumb_path))


def patch_reader(spatial):
    """Create a real image reader; never reconstruct RGB from feature vectors."""
    from PIL import Image
    source, resolve = spatial["source"], spatial["resolve"]
    mode = source.get("mode")
    if mode == "indexed_patch_images":
        path = resolve(source.get("indexed_patch_manifest"))
        if sha256(path) != source.get("indexed_patch_manifest_sha256"):
            raise ValueError("Indexed patch image manifest hash differs")
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        mapping = {int(r["patch_index"]): r for r in rows}
        if len(mapping) != len(rows):
            raise ValueError("Duplicate patch indices in image manifest")
        def read(i):
            row = mapping[int(i)]
            p = Path(row["image_path"])
            p = p if p.is_absolute() else path.parent / p
            if row.get("image_sha256") and sha256(p) != row["image_sha256"]:
                raise ValueError(f"Patch image hash differs at original row {i}")
            return Image.open(p).convert("RGB")
        return read, lambda: None
    if mode == "wsi_read_region":
        import openslide
        slide_path = resolve(spatial["asset"]["slide"].get("path"))
        if sha256(slide_path) != spatial["asset"]["slide"].get("sha256"):
            raise ValueError("Original slide hash differs")
        slide = openslide.OpenSlide(str(slide_path))
        if list(slide.dimensions) != spatial["asset"]["slide"]["level0_dimensions"]:
            slide.close(); raise ValueError("Actual slide dimensions differ")
        level = int(source["read_level"])
        if not np.isclose(slide.level_downsamples[level], float(source["level_downsample"])):
            slide.close(); raise ValueError("Actual extraction level downsample differs")
        def read(i):
            loc = spatial["coords"][int(i)]
            if not np.allclose(loc, np.round(loc)):
                raise ValueError("OpenSlide requires integer level-0 locations")
            return slide.read_region(tuple(np.round(loc).astype(int)), level,
                                     tuple(int(x) for x in source["read_size_at_level"])).convert("RGB")
        return read, slide.close
    raise ValueError("Choose wsi_read_region or indexed_patch_images in the assets manifest")
