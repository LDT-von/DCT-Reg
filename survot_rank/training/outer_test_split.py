#!/usr/bin/env python3
"""Generate the outer-test split files for DCT v3.13 §3.1 #5.

§3.3 of ``paper/V313_IMPLEMENTATION_PLAN.md`` requires:

1. The original five-fold ``val`` is fixed as the outer test set.
2. The original ``train`` is split 80/20 into ``inner_train`` and
   ``inner_val`` for checkpoint selection.
3. The split is stratified on the censorship (event vs censored) label
   so each subset has approximately the same event rate as the parent.
4. The split is reproducible across model seeds: a fixed ``split_seed``
   (default 3) drives the per-patient randomisation.

The output mirrors the existing ``splits/<which_splits>/<study>/fold_<N>.csv``
layout so the rest of the training stack can read it without code changes
beyond recognising the new columns (``inner_train``, ``inner_val``,
``outer_test``).

Run::

    PYTHONPATH=/data1/DCT-Reg python -m survot_rank.training.outer_test_split \
        --which_splits 5fold_uni2h \
        --outer_which_splits 5fold_uni2h_outer_test_seed3 \
        --split_seed 3 \
        --inner_val_fraction 0.20 \
        --studies blca,kirc \
        --data_path /data1/dataset_csv

The script is idempotent: re-running with the same seed produces identical
CSVs, so paper reproducibility does not depend on filesystem ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from typing import Iterable

import numpy as np
import pandas as pd


# Default outer-test column names (overridable via CLI for ad-hoc studies).
DEFAULT_OUTER_TEST = "outer_test"
DEFAULT_INNER_TRAIN = "inner_train"
DEFAULT_INNER_VAL = "inner_val"

# Default split configuration (overridable via CLI).
DEFAULT_SPLIT_SEED = 3
DEFAULT_INNER_VAL_FRACTION = 0.20


def stratified_inner_split(
    train_ids: list[str],
    event_mask: list[int],
    inner_val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratified split of the parent training cohort into inner_train
    and inner_val, preserving event/censored proportions.

    Args:
        train_ids: full patient ID list of the parent train cohort.
        event_mask: 1/0 indicator of *event* (vs censored) per patient.
        inner_val_fraction: fraction of patients routed to inner_val.
        seed: RNG seed — fixed for reproducibility across model seeds.

    Returns:
        (inner_train_ids, inner_val_ids) preserving the parent order.

    Patient IDs are sorted before seeding so that the deterministic
    partitioning is invariant under CSV row shuffling.  Per-stratum
    shuffling uses a per-stratum ``numpy.random.default_rng(seed)``,
    which means the outer seed fully determines the split.
    """
    if not 0.0 < inner_val_fraction < 1.0:
        raise ValueError(
            f"inner_val_fraction must be in (0, 1), got {inner_val_fraction}"
        )
    if len(train_ids) != len(event_mask):
        raise ValueError(
            f"train_ids and event_mask length mismatch: {len(train_ids)} vs "
            f"{len(event_mask)}"
        )
    train_ids = list(train_ids)
    event_mask = [int(e) for e in event_mask]

    # Sort patient IDs so the partitioning is invariant to CSV row order.
    paired = sorted(zip(train_ids, event_mask))
    event_patients = [pid for pid, e in paired if e == 1]
    censored_patients = [pid for pid, e in paired if e == 0]

    rng = np.random.default_rng(seed)
    rng.shuffle(event_patients)
    rng.shuffle(censored_patients)

    def take(lst: list[str], fraction: float) -> tuple[list[str], list[str]]:
        n_val = max(1, int(round(len(lst) * fraction)))
        # Cap to len-1 so each stratum contributes at least one patient
        # to inner_train (otherwise inner_train could be empty for a
        # stratum with very few patients).
        n_val = min(n_val, max(0, len(lst) - 1))
        return lst[n_val:], lst[:n_val]

    ev_train, ev_val = take(event_patients, inner_val_fraction)
    ce_train, ce_val = take(censored_patients, inner_val_fraction)

    inner_train_ids = sorted(ev_train + ce_train)
    inner_val_ids = sorted(ev_val + ce_val)
    return inner_train_ids, inner_val_ids


def build_outer_test_frame(
    fold_df: pd.DataFrame,
    inner_val_fraction: float,
    seed: int,
    *,
    train_col: str = "train",
    val_col: str = "val",
    inner_train_col: str = DEFAULT_INNER_TRAIN,
    inner_val_col: str = DEFAULT_INNER_VAL,
    outer_test_col: str = DEFAULT_OUTER_TEST,
) -> pd.DataFrame:
    """Build the three-column outer_test split frame for one fold.

    Args:
        fold_df: the parent split frame with ``train`` and ``val`` columns.
        inner_val_fraction: fraction of the parent train routed to inner_val.
        seed: per-fold deterministic seed — derived from the global seed and
            the fold index so different folds get independent shuffles.
        train_col: name of the parent train column in ``fold_df``.
        val_col: name of the parent val column in ``fold_df``.
        inner_train_col: name of the inner_train column in the output frame.
        inner_val_col: name of the inner_val column in the output frame.
        outer_test_col: name of the outer_test column in the output frame.

    Returns:
        A new ``pd.DataFrame`` with three columns ``inner_train``,
        ``inner_val``, ``outer_test``.  Rows are padded with ``NaN`` so the
        frame has length ``max(len(inner_train), len(inner_val),
        len(outer_test))`` — same convention as the parent split.

    The event/censored mask is *not* available from the parent split alone
    (the split CSV only stores patient IDs).  We therefore accept it via a
    dedicated hook — see :func:`build_outer_test_frame_with_clinical_df` —
    that looks up event information from the clinical frame.
    """
    raise NotImplementedError(
        "Use build_outer_test_frame_with_clinical_df — the split CSV alone "
        "does not carry the censorship label required for stratification."
    )


def build_outer_test_frame_with_clinical_df(
    fold_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    inner_val_fraction: float,
    seed: int,
    *,
    train_col: str = "train",
    val_col: str = "val",
    inner_train_col: str = DEFAULT_INNER_TRAIN,
    inner_val_col: str = DEFAULT_INNER_VAL,
    outer_test_col: str = DEFAULT_OUTER_TEST,
    case_id_col: str = "case id",
    censorship_var: str = "censorship",
) -> pd.DataFrame:
    """Build the three-column outer_test split frame, looking up event
    status from ``clinical_df`` to stratify the inner_train / inner_val
    split.

    Args:
        fold_df: the parent split frame with ``train`` and ``val`` columns.
        clinical_df: the DCT clinical frame (must contain ``case_id_col``
            and ``censorship_var``).
        inner_val_fraction: fraction of the parent train routed to inner_val.
        seed: per-fold deterministic seed.
        train_col: name of the parent train column.
        val_col: name of the parent val column.
        inner_train_col: output column name for inner_train patients.
        inner_val_col: output column name for inner_val patients.
        outer_test_col: output column name for outer_test patients.
        case_id_col: column name in ``clinical_df`` carrying patient IDs.
        censorship_var: column name in ``clinical_df`` carrying the
            censorship indicator (``1`` = censored, ``0`` = event in the
            DCT convention — see below).

    Returns:
        A new ``pd.DataFrame`` with the three new columns.

    Censorship convention in DCT clinical files: ``censorship=1`` means
    the patient is *censored* (the observation ended before the event);
    ``censorship=0`` means the patient experienced the event.  We use
    ``event = 1 - censorship`` for the stratum key.
    """
    train_ids = fold_df[train_col].dropna().astype(str).tolist()
    val_ids = fold_df[val_col].dropna().astype(str).tolist()

    clinical_lookup = clinical_df.set_index(case_id_col)
    missing = [pid for pid in train_ids if pid not in clinical_lookup.index]
    if missing:
        raise ValueError(
            f"clinical_df is missing {len(missing)} train patient IDs "
            f"(e.g. {missing[:3]}); cannot stratify without event labels"
        )

    event_mask = [
        1 - int(clinical_lookup.loc[pid, censorship_var]) for pid in train_ids
    ]

    inner_train_ids, inner_val_ids = stratified_inner_split(
        train_ids=train_ids,
        event_mask=event_mask,
        inner_val_fraction=inner_val_fraction,
        seed=seed,
    )

    return _make_outer_test_frame(
        inner_train_ids=inner_train_ids,
        inner_val_ids=inner_val_ids,
        outer_test_ids=val_ids,
        inner_train_col=inner_train_col,
        inner_val_col=inner_val_col,
        outer_test_col=outer_test_col,
    )


def _make_outer_test_frame(
    inner_train_ids: list[str],
    inner_val_ids: list[str],
    outer_test_ids: list[str],
    inner_train_col: str,
    inner_val_col: str,
    outer_test_col: str,
) -> pd.DataFrame:
    """Materialise the three-column DataFrame from three ID lists.

    Rows are padded with ``NaN`` so each column can hold its own length;
    CSV readers will treat NaN as missing, matching the parent layout.
    """
    n = max(len(inner_train_ids), len(inner_val_ids), len(outer_test_ids))
    inner_train_col_data = inner_train_ids + [None] * (n - len(inner_train_ids))
    inner_val_col_data = inner_val_ids + [None] * (n - len(inner_val_ids))
    outer_test_col_data = outer_test_ids + [None] * (n - len(outer_test_ids))
    return pd.DataFrame(
        {
            inner_train_col: inner_train_col_data,
            inner_val_col: inner_val_col_data,
            outer_test_col: outer_test_col_data,
        }
    )


def _fingerprint_outer_split(frame: pd.DataFrame) -> str:
    """Stable SHA-256 fingerprint of an outer-test split frame.

    Two generators that produce identical frames (same seed, same inputs)
    yield identical fingerprints.  Useful for paper reproducibility:
    ``git diff`` of generated CSVs is empty when nothing changed.
    """
    payload = "\n".join(
        ",".join(str(cell) for cell in row)
        for row in frame.itertuples(index=False, name=None)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_outer_test_splits(
    *,
    data_path: str,
    which_splits: str,
    outer_which_splits: str,
    studies: Iterable[str],
    folds: Iterable[int],
    inner_val_fraction: float,
    seed: int,
    clinical_csv_name: str = "clinical.csv",
) -> dict[str, dict[int, str]]:
    """Generate outer-test split files for every (study, fold) pair.

    Args:
        data_path: root containing ``splits/<which_splits>/<study>/``.
        which_splits: parent split directory name (e.g. ``5fold_uni2h``).
        outer_which_splits: output split directory name
            (e.g. ``5fold_uni2h_outer_test_seed3``).
        studies: iterable of study names (e.g. ``["blca", "kirc"]``).
        folds: iterable of fold indices (e.g. ``range(5)``).
        inner_val_fraction: fraction of the parent train routed to inner_val.
        seed: base seed; per-fold seed = ``seed + 1009 * fold`` for
            independence across folds.
        clinical_csv_name: name of the clinical CSV inside each study dir.

    Returns:
        Nested dict ``{study: {fold: fingerprint}}`` for audit logging.

    Side effects:
        Writes ``splits/<outer_which_splits>/<study>/fold_<N>.csv`` to disk.
    """
    fingerprints: dict[str, dict[int, str]] = {}
    for study in studies:
        clinical_path = os.path.join(data_path, study, clinical_csv_name)
        if not os.path.exists(clinical_path):
            raise FileNotFoundError(
                f"clinical CSV not found for study {study!r}: {clinical_path}"
            )
        clinical_df = pd.read_csv(clinical_path)

        out_study_dir = os.path.join(
            data_path, "splits", outer_which_splits, study
        )
        os.makedirs(out_study_dir, exist_ok=True)

        fingerprints[study] = {}
        for fold in folds:
            parent_path = os.path.join(
                data_path, "splits", which_splits, study, f"fold_{fold}.csv"
            )
            if not os.path.exists(parent_path):
                raise FileNotFoundError(
                    f"parent split not found: {parent_path}"
                )
            parent_df = pd.read_csv(parent_path)

            frame = build_outer_test_frame_with_clinical_df(
                fold_df=parent_df,
                clinical_df=clinical_df,
                inner_val_fraction=inner_val_fraction,
                seed=seed + 1009 * fold,
            )
            out_path = os.path.join(out_study_dir, f"fold_{fold}.csv")
            frame.to_csv(out_path, index=False)

            fingerprints[study][fold] = _fingerprint_outer_split(frame)
    return fingerprints


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate outer-test split CSVs for DCT v3.13 §3.1 #5.  "
            "Idempotent under the same seed."
        )
    )
    parser.add_argument("--data_path", default="/data1/dataset_csv")
    parser.add_argument("--which_splits", default="5fold_uni2h")
    parser.add_argument(
        "--outer_which_splits",
        default="5fold_uni2h_outer_test_seed3",
    )
    parser.add_argument("--studies", default="blca,kirc")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--inner_val_fraction",
        type=float,
        default=DEFAULT_INNER_VAL_FRACTION,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    studies = [s.strip() for s in args.studies.split(",") if s.strip()]
    folds = [int(f.strip()) for f in args.folds.split(",") if f.strip()]
    fingerprints = generate_outer_test_splits(
        data_path=args.data_path,
        which_splits=args.which_splits,
        outer_which_splits=args.outer_which_splits,
        studies=studies,
        folds=folds,
        inner_val_fraction=args.inner_val_fraction,
        seed=args.split_seed,
    )
    for study, fold_map in fingerprints.items():
        for fold, fp in fold_map.items():
            print(f"[outer-split] study={study} fold={fold} sha256={fp[:12]}")


if __name__ == "__main__":
    main()
