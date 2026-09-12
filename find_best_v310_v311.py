"""Find best v3.10 and v3.11 versions by per-fold val_cindex and 5-fold mean."""
import os
import csv
from collections import defaultdict

ROOT = "/data1/DCT-Reg/results"

def load_curve(path):
    """Return list of (epoch, val_cindex)."""
    out = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                out.append((int(row["epoch"]), float(row["val_cindex"])))
            except (KeyError, ValueError):
                continue
    return out

def best_per_fold(curve):
    """Return max val_cindex."""
    if not curve:
        return None
    return max(v for _, v in curve)

def collect_experiment(exp_dir):
    """Return (mean_best_cindex, per_fold_best, num_folds, fold_paths)."""
    curves = sorted([f for f in os.listdir(exp_dir) if f.startswith("epoch_curve_fold") and f.endswith(".csv")])
    fold_bests = []
    for cf in curves:
        curve = load_curve(os.path.join(exp_dir, cf))
        b = best_per_fold(curve)
        if b is not None:
            fold_bests.append(b)
    if not fold_bests:
        return None
    return sum(fold_bests) / len(fold_bests), fold_bests, len(fold_bests), curves

def main():
    candidates = []
    # v3.10 final (e30 brca, e50 brca, e50 kirc, blca under final_50ep_old)
    v310_dirs = [
        ("v3.10/final/brca/e30",
         f"{ROOT}/dct_v3.10/robust/final/brca/brca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_brca_50ep"),
        ("v3.10/final/brca/e50 (empty?)",
         f"{ROOT}/dct_v3.10/robust/final/brca/brca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_brca_50ep"),
        ("v3.10/final/kirc/e50",
         f"{ROOT}/dct_v3.10/robust/final/kirc/kirc/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_kirc_50ep"),
        ("v3.10/final_50ep_old/blca/e50",
         f"{ROOT}/dct_v3.10/robust/final_50ep_old/blca/blca/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_blca_50ep"),
        ("v3.10/final_50ep_old/hnsc/e50",
         f"{ROOT}/dct_v3.10/robust/final_50ep_old/hnsc/hnsc/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_hnsc_50ep"),
        ("v3.10/final_50ep_old/kirc/e50",
         f"{ROOT}/dct_v3.10/robust/final_50ep_old/kirc/kirc/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_kirc_50ep"),
        ("v3.10/final_50ep_old/lusc/e50",
         f"{ROOT}/dct_v3.10/robust/final_50ep_old/lusc/lusc/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_lusc_50ep"),
        ("v3.10/final_50ep_old/skcm/e50",
         f"{ROOT}/dct_v3.10/robust/final_50ep_old/skcm/skcm/SurvOTRank_dct_v310_directional_regularized_transport/0.0005_b8_survival_months_dss_Dim_256_e_50_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v310_dct_reg_skcm_50ep"),
    ]
    v311_dirs = [
        ("v3.11/blca/sdd_e30",
         f"{ROOT}/dct_v3.11/blca/blca/SurvOTRank_dct_v311_slot_disentanglement/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_sdd_blca_50ep"),
        ("v3.11/blca_uni_fixed/fold0",
         f"{ROOT}/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold0"),
        ("v3.11/blca_uni_fixed/fold1",
         f"{ROOT}/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold1"),
        ("v3.11/blca_uni_fixed/fold2",
         f"{ROOT}/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold2"),
        ("v3.11/blca_uni_fixed/fold3",
         f"{ROOT}/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold3"),
        ("v3.11/blca_uni_fixed/fold4",
         f"{ROOT}/dct_v311_blca_uni_fixed/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_fold4"),
    ]

    print("=" * 90)
    print("v3.10 candidates")
    print("=" * 90)
    for label, path in v310_dirs:
        if not os.path.isdir(path):
            print(f"{label}: MISSING")
            continue
        res = collect_experiment(path)
        if res is None:
            print(f"{label}: NO DATA")
            continue
        mean, folds, n, _ = res
        folds_str = ", ".join(f"{v:.4f}" for v in folds)
        print(f"{label}: 5fold_best_mean={mean:.4f}  (n_folds={n})  folds=[{folds_str}]")

    print()
    print("=" * 90)
    print("v3.11 candidates")
    print("=" * 90)
    for label, path in v311_dirs:
        if not os.path.isdir(path):
            print(f"{label}: MISSING")
            continue
        res = collect_experiment(path)
        if res is None:
            print(f"{label}: NO DATA")
            continue
        mean, folds, n, _ = res
        folds_str = ", ".join(f"{v:.4f}" for v in folds)
        print(f"{label}: 5fold_best_mean={mean:.4f}  (n_folds={n})  folds=[{folds_str}]")

if __name__ == "__main__":
    main()