"""Audit saved paired artifacts; never trains or runs patient inference."""
from pathlib import Path
import argparse
import csv
import json
import math
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from survot_rank.plugins.transport_only.slotspe_compare import ARMS, SCHEMA, sha, _score


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def predictions(path, manifest, fold, repeats):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(pid, r) for pid in fold["val"] for r in range(repeats)}
    keys = [(row["patient_id"], int(row["repeat"])) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, f"Prediction identities/repeats differ: {path}")
    for row in rows:
        pid = row["patient_id"]
        require(float(row["time"]) == manifest["patients"][pid]["time"] and
                float(row["censor"]) == manifest["patients"][pid]["censor"] and
                int(row["label"]) == fold["labels"][pid], "Prediction endpoints or labels differ")
        require(all(math.isfinite(float(row[k])) for k in ("risk", "logit_0", "logit_1", "logit_2", "logit_3")),
                "Nonfinite prediction")
        survival, risk = 1., 0.
        for k in range(4):
            value = float(row[f"logit_{k}"])
            # Numerically stable sigmoid complement.
            survival *= (math.exp(-value) / (1 + math.exp(-value))) if value >= 0 else 1 / (1 + math.exp(value))
            risk -= survival
        require(abs(risk - float(row["risk"])) < 2e-5, "Risk does not match saved logits")
    scores = []
    for repeat in range(repeats):
        values = [r for r in rows if int(r["repeat"]) == repeat]
        score = _score([float(r["censor"]) < 1 for r in values], [float(r["time"]) for r in values],
                       [float(r["risk"]) for r in values])
        require(score is not None, "Undefined patient metric")
        scores.append(score)
    return scores


def audit(root):
    root = Path(root)
    protocol, plan, status = read(root / "protocol.json"), read(root / "plan.json"), read(root / "status.json")
    require(plan["schema"] == SCHEMA and status["status"] == "completed_pending_independent_audit", "Batch incomplete/failed")
    require(all(protocol[k] == v for k, v in plan.items()), "Protocol diverged from locked plan")
    require(plan["host_version"] == "unrepaired_local_SlotSPE_snapshot" and plan["dct_added_losses"] == [], "Wrong host/objective")
    require(plan["selection"] == "fixed_final_epoch" and not plan["independent_test"], "Selection or evidence scope changed")
    require(sha(root / "prepared/manifest.json") == protocol["manifest_sha256"], "Manifest changed")
    require(sha(root / "shared_initial.pt") == protocol["shared_initial_sha256"], "Shared initial state changed")
    manifest = read(root / "prepared/manifest.json")
    cfg = plan["config"]
    fold = manifest["folds"][str(cfg["folds"][0])]
    require(manifest["source_hashes"] == plan["host_source_hashes"], "Prepared source differs")
    require(not set(fold["train"]) & set(fold["val"]), "Patient leakage across split")
    require(fold["binning"]["validation_outcomes_used_for_training_bins"] is False, "Validation-fitted bins")
    steps = math.ceil(len(fold["train"]) / cfg["batch_size"])
    summaries = {}
    for arm in ARMS:
        out = root / arm
        summary, curve = read(out / "summary.json"), read(out / "curve.json")
        require(summary["arm"] == arm and summary["epochs"] == cfg["epochs"] and len(curve) == cfg["epochs"], "Epoch budget mismatch")
        require([r["epoch"] for r in curve] == list(range(1, cfg["epochs"] + 1)), "Epoch sequence mismatch")
        require(summary["optimizer_steps"] == steps * cfg["epochs"], "Step budget mismatch")
        require(summary["checkpoint_sha256"] == sha(out / "last.pt"), "Checkpoint missing/changed")
        for row in curve:
            require(row["optimizer_steps"] == row["epoch"] * steps, "Incomplete optimizer steps")
            require(all(math.isfinite(row[k]) for k in ("train_main_nll", "train_auxiliary_total", "train_total", "train_seconds")), "Nonfinite curve")
            require(abs(row["train_total"] - row["train_main_nll"] - row["train_auxiliary_total"]) < 1e-4, "Objective does not equal host original")
        scores = predictions(out / "predictions.csv", manifest, fold, cfg["eval_repeats"])
        require(len(scores) == len(summary["cindex_by_repeat"]) and
                all(abs(a-b) < 1e-9 for a,b in zip(scores, summary["cindex_by_repeat"])), "C-index differs from predictions")
        require(abs(sum(scores) / len(scores) - summary["cindex_mean"]) < 1e-9, "Wrong mean metric")
        require(summary["train_seconds"] > 0 and math.isfinite(summary["train_seconds"]), "Invalid timing")
        if arm != "native":
            with (out / "transport.csv").open(encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
            expected = {(f"train_epoch_{e}", -1, pid) for e in range(1, cfg["epochs"]+1) for pid in fold["train"]}
            expected |= {("validation", r, pid) for r in range(cfg["eval_repeats"]) for pid in fold["val"]}
            if arm == "learned":
                expected |= {("same_weights_independent", r, pid) for r in range(cfg["eval_repeats"]) for pid in fold["val"]}
            keys = [(r["phase"], int(r["repeat"]), r["patient_id"]) for r in records]
            require(len(keys) == len(set(keys)) and set(keys) == expected, "Missing/duplicated transport telemetry")
            for row in records:
                require(all(math.isfinite(float(v)) for k,v in row.items() if k not in ("phase", "repeat", "patient_id")), "Nonfinite telemetry")
                require(abs(float(row["transport_mass"])-1) < 2e-4 and float(row["marginal_error_max"]) <= 1.1e-4, "Balanced marginal constraint failed")
                if arm == "independent" or row["phase"] == "same_weights_independent":
                    require(float(row["plan_independent_l1"]) < 2e-5, "Independent control retained coupling structure")
        if arm == "learned":
            controls = read(out / "same_weights_controls.json")
            for mode in ("independent", "bypass"):
                control_scores = predictions(out / f"same_weights_{mode}_predictions.csv", manifest, fold, cfg["eval_repeats"])
                require(len(control_scores) == len(controls[mode]["cindex_by_repeat"]) and
                        all(abs(a-b) < 1e-9 for a,b in zip(control_scores, controls[mode]["cindex_by_repeat"])), "Wrong same-weights metric")
        summaries[arm] = summary
    deltas = {"learned_minus_native": summaries["learned"]["cindex_mean"] - summaries["native"]["cindex_mean"],
              "learned_minus_independent": summaries["learned"]["cindex_mean"] - summaries["independent"]["cindex_mean"]}
    saved = read(root / "paired_summary.json")
    require(saved["summaries"] == summaries and all(abs(saved[k]-v) < 1e-9 for k,v in deltas.items()), "Paired summary mismatch")
    return {"status": "artifact_consistency_passed", **deltas,
        "extra_train_seconds": {arm: summaries[arm]["train_seconds"] - summaries["native"]["train_seconds"] for arm in ARMS[1:]},
        "scope": "one fold/seed validation only; no significance, external-test or OT-over-attention claim",
        "limitations": ["Checkpoint hashes verified; checkpoint predictions not independently rerun",
                       "Protocol records are self-reports until source/config/cache provenance is audited on server",
                       "Raw feature files have path/size/mtime provenance; selected cache contents are hashed"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), ensure_ascii=False, indent=2))
