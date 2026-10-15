"""Audit saved training CSVs without loading a model or changing source artifacts.

Examples (output directories must not already exist)::

    python scripts/audits/audit_saved_training_curves.py --summary-csv curves.csv \
        --output-dir fresh_audit --write-corrected-csv
    python scripts/audits/audit_saved_training_curves.py --fold-dir saved_run \
        --output-dir fresh_original_audit --patience 8 --warmup 5

The summary format is ``epoch,fold0,...,mean,std`` and contains validation
C-index only. Original ``epoch_curve_foldN.csv`` files preserve every column,
including train diagnostics. Population standard deviation (ddof=0) is the
default; ddof=1 is available explicitly. Printed precision is used to estimate
rounding error, capped by --rounding-atol (default 0.0001) per comparison.

Early stopping reproduces the trainer: checkpoints use strict improvement and
keep the earliest tie; the controller ignores warmup epochs entirely and uses
strict improvement by min_delta. Only observations before the stopping decision
can select its checkpoint. Full-curve best scores are separately labelled as
retrospective. All calculations use the saved, possibly rounded values.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import glob
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


INTERPRETATION_LIMITS = [
    "A decrease in a diversity penalty is not evidence of slot collapse; "
    "direct slot distances and similarities are required.",
    "Reference deficit is a transport budget quantity, not training prediction loss.",
    "A low deficit-weighted native loss does not establish native branch "
    "activation, predictive quality, or information preservation.",
    "A near-zero signed mean relation gap may hide positive and negative "
    "patient gaps; it is not mean absolute gap or a zero relation contribution.",
    "Loss values and even nonzero probe gaps cannot establish effective "
    "parameter gradients or a causal source of validation deterioration.",
    "Retrospective best validation scores are checkpoint-selection evidence, "
    "not held-out test performance or proof of generalization.",
]

EVIDENCE_GROUPS = {
    "training_prediction_loss": ["train_global_nll", "train_nll", "train_primary_nll",
                                 "train_optimization_main_nll", "train_eval_main_nll"],
    "training_cindex": ["train_cindex", "train_c_index", "train_eval_cindex"],
    "validation_prediction_loss": ["val_main_nll"],
    "direct_slot_distance": ["slot_distance", "slot_pair_distance", "slot_pair_dist"],
    "slot_cosine_similarity": ["slot_cosine", "slot_pair_cosine"],
    "slot_risk_response_variance": ["slot_risk_variance", "slot_hazard_variance", "risk_response_variance"],
    "actual_transport_mass": ["transport_mass", "relation_effective_mass"],
    "plan_distance_from_independent": ["plan_independent_l1", "plan_independent_distance",
                                       "plan_l1_to_independent", "plan_independence_l1_normalized"],
    "deployed_transport_gain": ["residual_gain", "deployment_gain"],
    "local_readout_response": ["local_readout_norm"],
    "signed_relation_gap": ["relation_probe_nll_gap"],
    "mean_absolute_patient_relation_gap": ["relation_probe_nll_gap_abs", "relation_probe_abs_nll_gap"],
    "weighted_relation_contribution": ["relation_weighted"],
    "effective_cost_gradient": ["relation_cost_grad_norm", "relation_gradient_norm", "relation_cost_gradient_norm_raw"],
}


def _number(value: str, *, context: str, allow_missing: bool = False) -> float | None:
    if not value.strip():
        if allow_missing:
            return None
        raise ValueError(f"Missing numeric value: {context}")
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value {value!r}: {context}") from exc
    if not math.isfinite(result):
        if allow_missing:
            return None
        raise ValueError(f"Nonfinite numeric value: {context}")
    return result


def _half_quantum(value: str) -> float:
    """Bound half a unit of the reported decimal precision, not a data CI."""
    try:
        return float(Decimal(1).scaleb(Decimal(value).as_tuple().exponent)) / 2
    except (InvalidOperation, ValueError, OverflowError):
        return 0.0


def _cell(value: str) -> Any:
    if not value.strip():
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    return number if math.isfinite(number) else None


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]], dict[str, Any]]:
    """Hash the exact bytes parsed, including their original encoding/newlines."""
    path = path.resolve(strict=True)
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    fields = reader.fieldnames
    if not fields or "epoch" not in fields:
        raise ValueError(f"CSV needs an epoch column: {path}")
    if len(set(fields)) != len(fields):
        raise ValueError(f"Duplicate column names: {path}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"CSV contains no data rows: {path}")
    epochs = []
    for line, row in enumerate(rows, start=2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed CSV row at {path}:{line}")
        number = _number(row["epoch"], context=f"{path}:{line}:epoch")
        if number < 0 or not number.is_integer():
            raise ValueError(f"Epoch must be a nonnegative integer at {path}:{line}")
        epochs.append(int(number))
    if len(set(epochs)) != len(epochs):
        raise ValueError(f"Duplicate epochs: {path}")
    if epochs != sorted(epochs):
        raise ValueError(f"Epochs must be increasing for sequential audit: {path}")
    source = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(),
              "bytes": len(payload), "columns": fields, "row_count": len(rows)}
    return fields, rows, source


def _mean_std(values: list[float], ddof: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = statistics.fmean(values)
    if len(values) <= ddof:
        return mean, None
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - ddof)
    return mean, math.sqrt(variance)


def _aggregate_check(raw: dict[str, str], field: str, recomputed: float | None,
                     fold_raw: list[str], *, ddof: int, atol: float,
                     rounding_atol: float) -> dict[str, Any]:
    reported = _number(raw.get(field, ""), context=field, allow_missing=True)
    if reported is None or recomputed is None:
        return {"field": field, "status": "unavailable", "reported": reported,
                "recomputed": recomputed, "reason": "missing aggregate or insufficient fold values"}
    input_errors = [_half_quantum(value) for value in fold_raw]
    if field == "mean":
        input_bound = statistics.fmean(input_errors) if input_errors else 0.0
    else:
        # Standard deviation is Lipschitz in the Euclidean centered vector.
        n = len(input_errors)
        input_bound = math.sqrt(math.fsum(err * err for err in input_errors) / (n - ddof))
    precision_bound = _half_quantum(raw[field]) + input_bound
    tolerance = atol + min(rounding_atol, precision_bound)
    difference = reported - recomputed
    return {"field": field, "status": "consistent_with_rounding" if abs(difference) <= tolerance else "inconsistent",
            "reported": reported, "recomputed": recomputed, "reported_minus_recomputed": difference,
            "absolute_difference": abs(difference), "tolerance": tolerance,
            "precision_error_bound": precision_bound}


def simulate_early_stopping(rows: list[dict[str, Any]], *, metric: str = "val_cindex",
                            patience: int = 5, warmup: int = 0, min_delta: float = 0.0,
                            mode: str = "max") -> dict[str, Any]:
    """Consume only observed rows. Full-run hindsight never enters this decision."""
    if patience < 0 or warmup < 0 or min_delta < 0 or not math.isfinite(min_delta):
        raise ValueError("patience, warmup and min_delta must be finite and nonnegative")
    if mode not in {"max", "min"}:
        raise ValueError("mode must be max or min")
    if not rows:
        raise ValueError("Cannot simulate an empty curve")
    epochs = [row["epoch"] for row in rows]
    if epochs != list(range(len(rows))):
        return {"status": "unavailable", "reason": "simulation needs contiguous epochs starting at zero"}
    if any(not isinstance(row.get(metric), (int, float)) or not math.isfinite(row[metric]) for row in rows):
        return {"status": "unavailable", "reason": f"missing finite {metric} observation"}
    sign = 1 if mode == "max" else -1
    checkpoint_best = controller_best = -math.inf
    checkpoint_epoch = None
    controller_best_epoch = None
    bad_epochs = 0
    trace = []
    triggered = False
    for row in rows:
        epoch, value = row["epoch"], row[metric]
        oriented = sign * value
        if oriented > checkpoint_best:
            checkpoint_best, checkpoint_epoch = oriented, epoch
        if patience > 0 and epoch >= warmup:
            if oriented > controller_best + min_delta:
                controller_best, controller_best_epoch = oriented, epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            triggered = bad_epochs >= patience
        trace.append({"epoch": epoch, "score": value, "bad_epochs": bad_epochs,
                      "controller_active": patience > 0 and epoch >= warmup,
                      "checkpoint_best_epoch": checkpoint_epoch})
        if triggered:
            break
    return {"status": "simulated", "metric": metric, "mode": mode,
            "patience": patience, "warmup": warmup, "min_delta": min_delta,
            "tie_rule": "earliest strict checkpoint improvement",
            "warmup_rule": "controller neither tracks best nor charges patience before warmup",
            "stopped_early": triggered, "stopped_epoch": trace[-1]["epoch"],
            "observed_epochs": len(trace), "checkpoint_best_epoch": checkpoint_epoch,
            "checkpoint_score": sign * checkpoint_best,
            "controller_best_epoch": controller_best_epoch,
            "controller_best_score": None if controller_best == -math.inf else sign * controller_best,
            "trace": trace}


def _best(rows: list[dict[str, Any]], metric: str, mode: str) -> dict[str, Any] | None:
    finite = [row for row in rows if isinstance(row.get(metric), (int, float)) and math.isfinite(row[metric])]
    if not finite:
        return None
    selected = (max if mode == "max" else min)(finite, key=lambda row: row[metric])
    return {"epoch": selected["epoch"], "score": selected[metric]}


def audit_curves(*, summary_csv: Path | None = None, fold_csvs: list[Path] | None = None,
                 metric: str = "val_cindex", ddof: int = 0, atol: float = 1e-10,
                 rounding_atol: float = 0.0001, patience: int = 5, warmup: int = 0,
                 min_delta: float = 0.0, mode: str = "max") -> dict[str, Any]:
    if bool(summary_csv) == bool(fold_csvs):
        raise ValueError("Supply exactly one summary CSV or a set of original fold CSVs")
    if ddof not in {0, 1} or atol < 0 or rounding_atol < 0 or not all(map(math.isfinite, [atol, rounding_atol])):
        raise ValueError("ddof must be 0/1 and tolerances finite and nonnegative")
    folds: dict[str, list[dict[str, Any]]] = {}
    sources = []
    aggregate_checks = []
    original_summary = []
    summary_epoch_raw: dict[int, dict[str, str]] = {}
    raw_folds: dict[int, list[str]] = {}
    if summary_csv:
        if metric != "val_cindex" or mode != "max":
            raise ValueError("Summary fold columns contain validation C-index only (metric=val_cindex, mode=max)")
        fields, raw, source = _read_csv(summary_csv)
        sources.append(source)
        names = sorted((name for name in fields if re.fullmatch(r"fold\d+", name)), key=lambda name: int(name[4:]))
        if not names:
            raise ValueError("Summary CSV needs fold0, fold1, ... columns")
        for name in names:
            folds[name] = []
        for row in raw:
            epoch = int(float(row["epoch"]))
            original_summary.append({**{name: _cell(value) for name, value in row.items()}, "epoch": epoch})
            summary_epoch_raw[epoch] = row
            raw_folds[epoch] = []
            for name in names:
                value = _number(row[name], context=f"epoch {epoch}:{name}", allow_missing=True)
                if value is not None and not 0 <= value <= 1:
                    raise ValueError(f"C-index out of [0,1] at epoch {epoch}:{name}")
                folds[name].append({"epoch": epoch, metric: value})
                if value is not None:
                    raw_folds[epoch].append(row[name])
    else:
        resolved = [Path(path).resolve(strict=True) for path in fold_csvs]
        if len(set(resolved)) != len(resolved):
            raise ValueError("Duplicate source files")
        for path in resolved:
            fields, raw, source = _read_csv(path)
            if metric not in fields:
                raise ValueError(f"Original fold CSV lacks selected metric {metric}: {path}")
            match = re.search(r"fold(\d+)(?:\.csv)?$", path.name)
            name = f"fold{int(match.group(1))}" if match else path.stem
            if name in folds:
                raise ValueError(f"Duplicate fold identity {name}; specify one run only")
            source["fold"] = name
            sources.append(source)
            rows = []
            for raw_row in raw:
                row = {name: _cell(value) for name, value in raw_row.items()}
                row["epoch"] = int(float(raw_row["epoch"]))
                row[metric] = _number(raw_row[metric], context=f"{path}:epoch {row['epoch']}:{metric}", allow_missing=True)
                if row[metric] is not None and "cindex" in metric and not 0 <= row[metric] <= 1:
                    raise ValueError(f"C-index out of [0,1] in {path}:epoch {row['epoch']}")
                rows.append(row)
            folds[name] = rows

    indexed = {name: {row["epoch"]: row for row in rows} for name, rows in folds.items()}
    epochs = sorted({epoch for rows in indexed.values() for epoch in rows})
    aggregates = []
    for epoch in epochs:
        row = {"epoch": epoch}
        for name, indexed_rows in indexed.items():
            row[name] = indexed_rows.get(epoch, {}).get(metric)
        values = [row[name] for name in folds if row[name] is not None]
        mean, std = _mean_std(values, ddof)
        row.update(mean=mean, std=std, n_folds=len(values), complete_fold_coverage=len(values) == len(folds))
        aggregates.append(row)
        if summary_csv:
            for field, calculated in (("mean", mean), ("std", std)):
                aggregate_checks.append({"epoch": epoch, **_aggregate_check(
                    summary_epoch_raw[epoch], field, calculated, raw_folds[epoch],
                    ddof=ddof, atol=atol, rounding_atol=rounding_atol)})

    per_fold = {}
    for name, rows in folds.items():
        selected = _best(rows, metric, mode)
        simulation = simulate_early_stopping(rows, metric=metric, patience=patience,
                                              warmup=warmup, min_delta=min_delta, mode=mode)
        if simulation["status"] == "simulated" and selected:
            sign = 1 if mode == "max" else -1
            simulation["missed_later_improvement"] = sign * selected["score"] > sign * simulation["checkpoint_score"]
            simulation["retrospective_best_available_if_training_continued"] = selected
        per_fold[name] = {"retrospective_selected_best": selected, "early_stopping": simulation,
                          "available_columns": sorted({key for row in rows for key in row}),
                          "rows": rows}
    selected_values = [data["retrospective_selected_best"]["score"] for data in per_fold.values()
                       if data["retrospective_selected_best"] is not None]
    best_mean, best_std = _mean_std(selected_values, ddof)
    columns = sorted({key for rows in folds.values() for row in rows for key in row})
    evidence = {}
    for group, patterns in EVIDENCE_GROUPS.items():
        found = [column for column in columns if any(pattern in column for pattern in patterns)]
        usable = [column for column in found if any(isinstance(row.get(column), (int, float)) and math.isfinite(row[column])
                                                   for rows in folds.values() for row in rows)]
        evidence[group] = {"status": "recorded" if usable else "missing", "columns": usable}
    signed_columns = [column for column in evidence["signed_relation_gap"]["columns"]
                      if "abs" not in column]
    evidence["signed_relation_gap"] = {"status": "recorded" if signed_columns else "missing", "columns": signed_columns}
    gap_summaries = []
    for name, rows in folds.items():
        for column in signed_columns:
            values = [row[column] for row in rows if isinstance(row.get(column), (int, float))]
            if values:
                gap_summaries.append({"fold": name, "column": column, "count": len(values),
                                      "minimum_signed_epoch_mean": min(values), "maximum_signed_epoch_mean": max(values),
                                      "maximum_absolute_epoch_mean": max(map(abs, values)),
                                      "near_zero_epoch_means_at_1e_8": sum(abs(value) <= 1e-8 for value in values),
                                      "mean_absolute_patient_gap_recoverable": False})
    inconsistencies = [check for check in aggregate_checks if check["status"] == "inconsistent"]
    complete = [row for row in aggregates if row["complete_fold_coverage"]]
    return {
        "schema": "saved_training_curve_audit_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "read-only saved CSV arithmetic; no models, patient processing, training or inference",
        "source_kind": "validation_cindex_summary_only" if summary_csv else "original_per_fold_epoch_curves",
        "sources": sources, "metric": metric, "mode": mode,
        "std_convention": {"ddof": ddof, "label": "population" if ddof == 0 else "sample",
                           "undefined_if_n_le_ddof": True},
        "rounding_policy": {"numeric_atol": atol, "maximum_rounding_atol": rounding_atol,
                            "rule": "atol + min(rounding_atol, aggregate half-quantum + propagated input precision)"},
        "aggregate_validation": {"inconsistent_count": len(inconsistencies), "inconsistencies": inconsistencies,
                                 "checks": aggregate_checks},
        "corrected_aggregates_from_available_folds": aggregates,
        "global_mean_peak_complete_coverage": _best(complete, "mean", mode),
        "independently_selected_per_fold_best": {"mean": best_mean, "std": best_std,
                                                "n_folds": len(selected_values),
                                                "claim": "retrospective validation selection; separate epoch per fold"},
        "per_fold": per_fold, "original_summary_rows": original_summary,
        "evidence_availability": evidence, "missing_evidence": [key for key, value in evidence.items() if value["status"] == "missing"],
        "signed_relation_gap_summary": gap_summaries,
        "mechanistic_cause_established": False, "interpretation_limits": INTERPRETATION_LIMITS,
        "coverage_warnings": [f"epoch {row['epoch']}: only {row['n_folds']}/{len(folds)} folds available"
                              for row in aggregates if not row["complete_fold_coverage"]],
    }


def write_outputs(report: dict[str, Any], output_dir: Path, *, corrected_csv: bool = False) -> dict[str, str]:
    """Claim a brand-new directory; never overwrite an existing directory/file."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report_path = output_dir / "saved_curve_audit.json"
    with report_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    paths = {"report": str(report_path)}
    if corrected_csv:
        csv_path = output_dir / "corrected_metric_aggregate.csv"
        rows = report["corrected_aggregates_from_available_folds"]
        fields = ["epoch", *report["per_fold"], "mean", "std", "n_folds", "complete_fold_coverage"]
        with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        paths["corrected_aggregate_csv"] = str(csv_path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--summary-csv", type=Path)
    inputs.add_argument("--fold-csv", type=Path, nargs="+", help="explicit original per-fold files from one run")
    inputs.add_argument("--fold-glob", help="quoted glob of original epoch_curve_foldN.csv files")
    inputs.add_argument("--fold-dir", type=Path, help="directory containing epoch_curve_foldN.csv (nonrecursive)")
    parser.add_argument("--output-dir", type=Path, required=True, help="fresh output directory; existing directory is rejected")
    parser.add_argument("--write-corrected-csv", action="store_true")
    parser.add_argument("--metric", default="val_cindex")
    parser.add_argument("--mode", choices=("max", "min"), default="max")
    parser.add_argument("--ddof", type=int, choices=(0, 1), default=0)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--rounding-atol", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    args = parser.parse_args(argv)
    fold_paths = args.fold_csv
    if args.fold_glob:
        fold_paths = [Path(path) for path in sorted(glob.glob(args.fold_glob))]
    if args.fold_dir:
        fold_paths = sorted(args.fold_dir.glob("epoch_curve_fold*.csv"))
    if not args.summary_csv and not fold_paths:
        parser.error("No original per-fold CSVs matched; no output was written")
    try:
        report = audit_curves(summary_csv=args.summary_csv, fold_csvs=fold_paths, metric=args.metric,
                              ddof=args.ddof, atol=args.atol, rounding_atol=args.rounding_atol,
                              patience=args.patience, warmup=args.warmup, min_delta=args.min_delta, mode=args.mode)
        paths = write_outputs(report, args.output_dir, corrected_csv=args.write_corrected_csv)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"outputs": paths, "aggregate_inconsistencies": report["aggregate_validation"]["inconsistent_count"],
                      "mechanistic_cause_established": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
