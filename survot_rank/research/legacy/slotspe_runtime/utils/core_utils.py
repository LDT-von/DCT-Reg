#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared DCT data-flow and survival-metric utilities.

This compatibility module contains only the dataset forwarding and metric
helpers used by the current DCT training and audit paths. The obsolete SlotSPE
model initializer and standalone training loop have been removed.
"""

import numpy as np
from sksurv.metrics import (
    brier_score,
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.util import Surv
import torch


def _extract_survival_metadata(dataset_factory, label_df=None):
    """Build a scikit-survival reference from one explicitly scoped cohort."""
    source_df = dataset_factory.clinical_df if label_df is None else label_df
    all_censorships = source_df[dataset_factory.censorship_var].to_numpy().flatten()
    all_event_times = source_df[dataset_factory.label_col].to_numpy().flatten()
    return Surv.from_arrays(
        event=(1 - all_censorships).astype(bool),
        time=all_event_times,
    )


def _unpack_data(data, device, omics_format):
    """Move one batch to device and normalize its modality layout."""
    data_wsi = data[0].to(device)
    if omics_format in ("Pathways", "RankedGenes"):
        data_omics = []
        for batch_index, omic_group in enumerate(data[1]):
            for omic_index, omic in enumerate(omic_group):
                omic = omic.to(device).unsqueeze(0)
                if batch_index == 0:
                    data_omics.append(omic)
                else:
                    data_omics[omic_index] = torch.cat(
                        (data_omics[omic_index], omic), dim=0
                    )
    else:
        data_omics = data[1].to(device)

    y_disc = data[2].to(device)
    event_time = data[3].to(device)
    censorship = data[4].to(device)
    clinical = data[5].to(device) if len(data) > 5 else None
    return data_wsi, data_omics, y_disc, event_time, censorship, clinical


def _process_data_and_forward(args, model, data, device, test=False):
    """Build model inputs for one DCT batch and execute the model."""
    data_wsi, data_omics, y_disc, event_time, censorship, clinical = _unpack_data(
        data, device, args.rna_format
    )
    input_args = {
        "x_wsi": data_wsi,
        "cur_epoch": args.cur_epoch,
        "wsi_missing": getattr(args, "wsi_missing", False) if test else False,
        "omic_missing": getattr(args, "omic_missing", False) if test else False,
        "event_time": event_time,
        "y": None if test else y_disc,
        "c": None if test else censorship,
    }
    if getattr(args, "otehv2v2_use_clinical", False) and clinical is not None:
        input_args["x_clinical"] = clinical

    if args.rna_format in ("Pathways", "RankedGenes"):
        input_args.update(
            {f"x_omic{index}": omic for index, omic in enumerate(data_omics, start=1)}
        )
    else:
        input_args["x_omics"] = data_omics
    return model(**input_args), y_disc, event_time, censorship


def _calculate_risk(hazards):
    """Convert discrete hazards into the repository's scalar risk convention."""
    survival = torch.cumprod(1 - torch.sigmoid(hazards), dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()


def _update_arrays(all_risk_scores, all_censorships, all_event_times, event_time, censor, risk, _):
    """Append one batch of evaluation outputs."""
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    return all_risk_scores, all_censorships, all_event_times


def _record_metric_error(metric_name, error, metric_error_path=None):
    """Report an unavailable metric without replacing it with a misleading zero."""
    message = f"[metrics] {metric_name} unavailable: {type(error).__name__}: {error}"
    print(message)
    if metric_error_path is not None:
        try:
            with open(metric_error_path, "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except OSError as write_error:
            print(f"[metrics] could not write diagnostics: {write_error}")


def _select_valid_metric_time_grid(event_times, survival_train, bins, survival_predictions):
    """Return evaluation times shared by the train and validation follow-up ranges."""
    event_times = np.asarray(event_times, dtype=float).reshape(-1)
    predictions = np.asarray(survival_predictions, dtype=float)
    bins = np.asarray(bins, dtype=float).reshape(-1)
    if predictions.ndim != 2 or predictions.shape[0] != event_times.size:
        raise ValueError("survival prediction rows must match validation outcomes")
    if event_times.size < 2:
        raise ValueError("at least two validation outcomes are required")
    num_columns = predictions.shape[1]
    if num_columns < 2 or bins.size < num_columns + 1:
        raise ValueError("discrete survival bins do not match prediction columns")

    candidate_times = np.concatenate((
        [event_times.min() + 1e-4],
        bins[1:num_columns - 1],
        [event_times.max() - 1e-4],
    ))
    train_times = np.asarray(survival_train["time"], dtype=float)
    lower = max(float(event_times.min()), float(train_times.min()))
    upper = min(float(event_times.max()), float(train_times.max()))
    valid = (
        np.isfinite(candidate_times)
        & (candidate_times >= lower)
        & (candidate_times < upper)
    )
    times = candidate_times[valid]
    predictions = predictions[:, valid]
    if times.size == 0:
        raise ValueError("no evaluation time lies inside both follow-up ranges")
    if np.any(np.diff(times) <= 0):
        raise ValueError("evaluation times must be strictly increasing")
    return times, predictions


def _calculate_metrics(
    loader,
    dataset_factory,
    survival_train,
    all_risk_scores,
    all_censorships,
    all_event_times,
    all_risk_by_bin_scores,
    metric_error_path=None,
):
    """Calculate survival metrics without silently converting failures to scores."""
    del loader
    risk = np.asarray(all_risk_scores)
    censorship = np.asarray(all_censorships)
    event_times = np.asarray(all_event_times)
    survival_predictions = np.asarray(all_risk_by_bin_scores)
    finite = (
        np.isfinite(risk)
        & np.isfinite(censorship)
        & np.isfinite(event_times)
        & np.isfinite(survival_predictions).all(axis=1)
    )
    risk, censorship, event_times = risk[finite], censorship[finite], event_times[finite]
    survival_predictions = survival_predictions[finite]
    if risk.size == 0:
        return 0.5, np.nan, np.nan, np.nan, np.nan

    try:
        c_index = concordance_index_censored(
            (1 - censorship).astype(bool), event_times, risk, tied_tol=1e-08
        )[0]
    except (ValueError, FloatingPointError) as error:
        _record_metric_error("cindex", error, metric_error_path)
        c_index = 0.5

    try:
        survival_test = Surv.from_arrays(
            event=(1 - censorship).astype(bool), time=event_times
        )
    except Exception as error:
        _record_metric_error("survival_test", error, metric_error_path)
        return c_index, np.nan, np.nan, np.nan, np.nan

    c_index_ipcw = bs = ibs = iauc = np.nan
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, risk)[0]
    except Exception as error:
        _record_metric_error("cindex_ipcw", error, metric_error_path)

    try:
        times, predictions = _select_valid_metric_time_grid(
            event_times, survival_train, dataset_factory.bins, survival_predictions
        )
    except Exception as error:
        _record_metric_error("time_grid", error, metric_error_path)
        return c_index, c_index_ipcw, bs, ibs, iauc

    try:
        _, bs = brier_score(survival_train, survival_test, predictions, times)
        if times.size >= 2:
            ibs = integrated_brier_score(survival_train, survival_test, predictions, times)
            _, iauc = cumulative_dynamic_auc(
                survival_train, survival_test, 1 - predictions[:, 1:], times[1:]
            )
    except Exception as error:
        _record_metric_error("time_dependent_metrics", error, metric_error_path)
    return c_index, c_index_ipcw, bs, ibs, iauc
