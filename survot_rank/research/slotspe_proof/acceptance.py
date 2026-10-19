"""Offline audit of returned original artifacts, without torch or server paths."""
import csv
import hashlib
import json
import math
from pathlib import Path
import zipfile


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _inside(root, relative):
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Returned artifact path escapes batch: {relative}")
    return path


def audit_return(root, delivery=None):
    """Fail closed on incomplete/tampered outputs; observed is not required."""
    root = Path(root).resolve()
    manifest = _read(root / "queue_manifest.json")
    config = _read(root / "resolved_config.json")
    status = _read(root / "queue_status.json")
    if config != manifest["config"]:
        raise ValueError("Saved configuration differs from queue manifest")
    if status.get("status") != "completed" or status.get("run_identity") != manifest["run_identity"]:
        raise ValueError("Queue incomplete or run identity differs; no server proof accepted")
    verified = set()
    for stage in manifest["stages"]:
        receipt = _read(_inside(root, "receipts/" + stage["id"] + ".json"))
        if receipt.get("run_identity") != manifest["run_identity"] or receipt.get("stage") != stage or not receipt.get("outputs"):
            raise ValueError(f"Invalid receipt: {stage['id']}")
        for relative, digest in receipt["outputs"].items():
            file = _inside(root, relative)
            if not file.is_file() or _sha(file) != digest:
                raise ValueError(f"Artifact missing or hash differs: {relative}")
            verified.add(file.relative_to(root).as_posix())
    implementation_verified = False
    if delivery:
        with zipfile.ZipFile(delivery) as archive:
            names = [name for name in archive.namelist() if name.endswith("/DELIVERY_MANIFEST.json")]
            if len(names) != 1:
                raise ValueError("Delivery manifest is missing or ambiguous")
            delivered = json.loads(archive.read(names[0]))["files_sha256"]
        implementation = manifest.get("implementation_sha256")
        if not implementation:
            raise ValueError("Legacy batch lacks full implementation map; audit without --delivery and retain that limitation")
        for name, digest in implementation.items():
            if delivered.get("survot_rank/research/slotspe_proof/" + name) != digest:
                raise ValueError(f"Returned implementation differs from supplied delivery: {name}")
        implementation_verified = True
    summary = _read(root / "proof_summary.json")
    if summary.get("status") != "completed" or summary.get("transport_added") is not False:
        raise ValueError("Invalid proof summary")
    if manifest["checks_only"]:
        raise ValueError("Only synthetic/metadata checks were run; no server patient proof accepted")
    prepared = _read(root / "prepared/manifest.json")
    arms = ["A", "B", "C"] if config.get("experiment") == "attribution_abc" else ["original", "readout_fix"]
    revised = manifest.get("artifact_contract") == "slotspe_abc_patient_intervention_v2"
    gaps, pairs = [], []

    def require_hashed(folder, names):
        for name in names:
            relative = (folder / name).relative_to(root).as_posix()
            if relative not in verified:
                raise ValueError(f"Required artifact is not covered by a completion receipt: {relative}")

    require_hashed(root, ["proof_summary.json", "proof_summary.txt", "patient_summary.json", "prepared/manifest.json"])
    patients_summary = _read(root / "patient_summary.json")
    if patients_summary.get("status") != "completed" or summary.get("patients") != patients_summary:
        raise ValueError("Patient/proof summary differs or is incomplete")

    def prediction_audit(folder, val_ids, labels):
        repeats = _read(folder / "eval_repeats.json")
        if len(repeats) != config["eval_repeats"]:
            raise ValueError("Wrong repeated-evaluation budget")
        with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if [row["patient_id"] for row in rows] != val_ids:
            raise ValueError("Prediction patient identities/order differ")
        for i, row in enumerate(rows):
            patient = prepared["patients"][val_ids[i]]
            if (float(row["time"]) != patient["time"] or float(row["censor"]) != patient["censor"] or int(row["label"]) != labels[val_ids[i]]):
                raise ValueError("Prediction endpoint/label differs from patient manifest")
            for j, repeat in enumerate(repeats):
                if len(repeat["risks"]) != len(val_ids) or not math.isfinite(float(repeat["risks"][i])) or float(row[f"risk_repeat_{j:02d}"]) != repeat["risks"][i]:
                    raise ValueError("CSV/JSON prediction differs or is nonfinite")

    for fold in config["folds"]:
        data = prepared["folds"][str(fold)]
        if set(data["train"]) & set(data["val"]):
            raise ValueError("Patient split overlap")
        if data["binning"]["validation_outcomes_used_for_training_bins"] is not False:
            raise ValueError("Comparable training did not use train-fold binning")
        for seed in config["seeds"]:
            folder = root / "pairs" / f"fold_{fold}" / f"seed_{seed}"
            require_hashed(folder, ["protocol.json", "summary.json", "shared_initial_state.pt"])
            protocol = _read(folder / "protocol.json")
            pair = _read(folder / "summary.json")
            if pair.get("status") != "completed" or protocol["arms"] != arms or [a["arm"] for a in pair["arms"]] != arms:
                raise ValueError("Incomplete or incorrect training arms")
            for key in ("epochs", "patches", "batch_size", "lr", "eval_repeats"):
                if protocol[key] != config[key]:
                    raise ValueError(f"Protocol/config mismatch: {key}")
            if protocol["manifest_sha256"] != _sha(root / "prepared/manifest.json"):
                raise ValueError("Prepared patient manifest hash differs")
            if protocol["source_hashes_before"] != protocol["source_hashes_after"]:
                raise ValueError("Source changed during training")
            if manifest.get("implementation_sha256") and any(protocol[key] != manifest["implementation_sha256"][name] for key, name in (("runner_sha256", "patient.py"), ("model_controls_sha256", "model.py"))):
                raise ValueError("Protocol implementation differs from queue implementation")
            expected_steps = math.ceil(len(data["train"]) / config["batch_size"]) * config["epochs"]
            initial = _sha(folder / "shared_initial_state.pt")
            for arm in pair["arms"]:
                arm_root = folder / arm["arm"]
                require_hashed(arm_root, ["summary.json", "curve.json", "last.pt", "trained_selector_gradients.json", "eval_repeats.json", "validation_predictions.csv"])
                if _read(arm_root / "summary.json") != arm:
                    raise ValueError("Per-arm and pair summaries differ")
                if arm["optimizer_steps"] != expected_steps or arm["same_initial_state_sha256"] != initial or arm["checkpoint_sha256"] != _sha(arm_root / "last.pt"):
                    raise ValueError("Training budget/checkpoint/initial-state mismatch")
                curve = _read(arm_root / "curve.json")
                if len(curve) != config["epochs"] or [r["epoch"] for r in curve] != list(range(1, config["epochs"] + 1)) or curve[-1]["optimizer_steps"] != expected_steps:
                    raise ValueError("Incomplete learning curve")
                gradients = _read(arm_root / "trained_selector_gradients.json")
                if not gradients["parameters"] or not set(gradients["patient_ids"]).issubset(data["train"]):
                    raise ValueError("Selector gradient provenance missing")
                prediction_audit(arm_root, data["val"], data["labels"])
                probe_path = arm_root / "patient_slot_intervention.json"
                if revised:
                    require_hashed(arm_root, ["patient_slot_intervention.json"])
                    probe = _read(probe_path)
                    records = [r for b in probe["batches"] for r in b["records"]]
                    if probe["patient_ids"] != data["val"] or [r["patient_id"] for r in records] != data["val"] or probe["checkpoint_sha256"] != arm["checkpoint_sha256"]:
                        raise ValueError("Patient slot intervention provenance differs")
                elif not probe_path.exists():
                    gaps.append(f"fold{fold}/seed{seed}/{arm['arm']}: real patient final-slot intervention absent in legacy package")
            if "original" in arms:
                same = pair["same_weights_eval_fix"]
                require_hashed(folder / "same_weights_eval_fix", ["summary.json", "eval_repeats.json", "validation_predictions.csv"])
                if same["new_optimizer_steps"] != 0 or same["checkpoint_sha256"] != pair["arms"][0]["checkpoint_sha256"]:
                    raise ValueError("Same-weights evaluation provenance differs")
                prediction_audit(folder / "same_weights_eval_fix", data["val"], data["labels"])
            else:
                require_hashed(folder, ["matched_state_bc_checks.json"])
                matched = _read(folder / "matched_state_bc_checks.json")
                if not all(v for k, v in matched["checks"].items() if k != "C_main_nonzero_each_modality"):
                    raise ValueError("Matched B/C structural contract failed")
            pairs.append({"fold": fold, "seed": seed, "arms": arms, "checks": pair["checks"]})
    if patients_summary.get("completed_pairs") != len(pairs) or len(patients_summary.get("results", [])) != len(pairs):
        raise ValueError("Patient-summary pair count differs")
    return {"status": "artifact_contract_passed", "root": str(root), "verified_hashed_files": len(verified),
        "implementation_matches_delivery": implementation_verified, "config": config, "pairs": pairs,
        "evidence_gaps": gaps, "evidence_scope": "returned_artifact_integrity_and_protocol",
        "patient_benefit": "requires_paired_result_interpretation", "attribution_conclusion": "not_automatic",
        "limitation": "Artifact integrity does not independently attest remote execution or source data truth; inspect original logs/environment and scientific observations"}
