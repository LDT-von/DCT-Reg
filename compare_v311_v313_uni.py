"""Compare v3.11 uni vs v3.13 uni across all cancers (5-fold split).

Sources:
- v3.11 (uni, slot_interpretable):
  * BLCA    : dct_v311_blca_uni_fixed/blca/blca/.../dct_v311_blca_uni/
  * pancancer: dct_v311_blca_uni_fixed_pancancer/<cancer>/<cancer>/.../dct_v311_blca_uni_<cancer>_fold{0..4}_fixed/

- v3.13 (uni, transport_reconstruction):
  * BLCA    : dct_v313_blca_uni/blca/blca/.../dct_v313_blca_uni/  (currently 1 epoch only → smoke)
  * pancancer: dct_v313_<cancer>_uni/blca/<cancer>/.../dct_v313_brca_uni/

Best val_cindex per fold is taken from epoch_curve_fold{0..4}.csv (max across epochs).
"""
import os
import csv
import statistics

ROOT = "/data1/DCT-Reg/results"


def load_best_from_curve(path):
    """Return (best_val_cindex, best_epoch) from an epoch_curve CSV."""
    best_c, best_ep = -1, -1
    best_row = None
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                c = float(row["val_cindex"])
                e = int(row["epoch"])
                if c > best_c:
                    best_c, best_ep = c, e
                    best_row = row
            except (KeyError, ValueError, TypeError):
                continue
    if best_row is None:
        return None
    def sf(key, default=""):
        v = best_row.get(key, default)
        try:
            return float(v)
        except (ValueError, TypeError):
            return default
    return {
        "epoch": best_ep,
        "val_cindex": best_c,
        "val_cindex_ipcw": sf("val_cindex_ipcw", -1),
        "val_IBS": sf("val_IBS", 1e9),
        "val_iauc": sf("val_iauc", -1),
    }


def collect_folds_5(base_dir):
    """Collect best per fold from base_dir/epoch_curve_fold{0..4}.csv."""
    results = []
    for fold in range(5):
        path = os.path.join(base_dir, f"epoch_curve_fold{fold}.csv")
        if os.path.isfile(path):
            results.append(load_best_from_curve(path))
    return results


def collect_v311(cancer):
    if cancer == "blca":
        d = (f"{ROOT}/dct_v311_blca_uni_fixed/blca/blca/"
             f"SurvOTRank_dct_v311_slot_interpretable/"
             f"0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni")
        return collect_folds_5(d)
    # pancancer: each fold is a separate single-fold run
    folds = []
    for fold in range(5):
        d = (f"{ROOT}/dct_v311_blca_uni_fixed_pancancer/{cancer}/{cancer}/"
             f"SurvOTRank_dct_v311_slot_interpretable/"
             f"0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_{cancer}_fold{fold}_fixed")
        if os.path.isdir(d):
            p = os.path.join(d, f"epoch_curve_fold{fold}.csv")
            if os.path.isfile(p):
                folds.append(load_best_from_curve(p))
    return folds


def collect_v313(cancer):
    if cancer == "blca":
        d = (f"{ROOT}/dct_v313_blca_uni/blca/blca/"
             f"SurvOTRank_dct_v313_transport_reconstruction/"
             f"0.0005_b32_survival_months_dss_Dim_256_e_1_g_Pathways_sig_combine_seed3_rW_16_rG_16_sp_dct_v313_blca_uni")
    else:
        d = (f"{ROOT}/dct_v313_{cancer}_uni/blca/{cancer}/"
             f"SurvOTRank_dct_v313_transport_reconstruction/"
             f"0.0005_b32_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_16_rG_16_sp_dct_v313_brca_uni")
    return collect_folds_5(d)


def fmt(v):
    return f"{v:.4f}" if v is not None else "  --  "


def mean_std(vals):
    if not vals:
        return None, None
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    cancers = ["blca", "brca", "coadread", "hnsc", "kirc", "luad", "lusc", "skcm"]
    summary = {}

    print("=" * 130)
    print("v3.11 uni vs v3.13 uni  |  5-fold CV  |  metric = best epoch val_cindex  (epoch@best in parens)")
    print("=" * 130)
    print(f"{'cancer':<10}{'method':<8}{'fold0':<14}{'fold1':<14}{'fold2':<14}{'fold3':<14}{'fold4':<14}{'mean±std':<16}{'n'}")
    print("-" * 130)

    for cancer in cancers:
        v311 = collect_v311(cancer)
        v313 = collect_v313(cancer)
        for label, folds in [("v3.11", v311), ("v3.13", v313)]:
            cidx = [f["val_cindex"] for f in folds if f]
            epc  = [f["epoch"] for f in folds if f]
            n    = len(cidx)
            m, s = mean_std(cidx)

            def fv(i):
                if i < n:
                    return f"{cidx[i]:.4f}(ep{epc[i]})"
                return "  --  "
            row = [cancer, label, fv(0), fv(1), fv(2), fv(3), fv(4),
                   f"{m:.4f}±{s:.4f}" if m is not None else "   N/A   ", n]
            print(f"{row[0]:<10}{row[1]:<8}{row[2]:<14}{row[3]:<14}{row[4]:<14}{row[5]:<14}{row[6]:<14}{row[7]:<16}{row[8]}")
            summary[(cancer, label)] = (m, s, n)

    print()
    print("=" * 120)
    print("Compact Summary  |  val_cindex mean ± std  (n = number of folds)")
    print("=" * 120)
    print(f"{'cancer':<10}{'v3.11 mean±std (n)':<24}{'v3.13 mean±std (n)':<24}{'Δ(v313-v311)':<15}{'note'}")
    print("-" * 120)

    # also print detailed per-metric table
    print()
    print("=" * 150)
    print("Detailed: val_cindex (c-index), val_cindex_ipcw (ipcw-cindex), val_IBS (IBS, lower=better), val_iauc (iauc)")
    print("=" * 150)
    print(f"{'cancer':<10}{'method':<8}{'c-index fold0':<14}{'ipcw fold0':<14}{'IBS fold0':<14}{'iauc fold0':<14}  ...")
    print("-" * 150)
    for cancer in cancers:
        v311 = collect_v311(cancer)
        v313 = collect_v313(cancer)
        for label, folds in [("v3.11", v311), ("v3.13", v313)]:
            def fv0(fld, i):
                if i < len(folds) and folds[i]:
                    v = folds[i][fld]
                    return f"{v:.4f}"
                return "  --  "
            row = [cancer, label] + [fv0("val_cindex", i) for i in range(5)] + \
                  [fv0("val_cindex_ipcw", i) for i in range(5)]
            print(f"{row[0]:<10}{row[1]:<8}" + "".join(f"{v:<14}" for v in row[2:]))

    # mean summary
    print()
    print("=" * 110)
    print("Mean C-Index comparison  (val_cindex)")
    print("=" * 110)
    print(f"{'cancer':<10}{'v3.11':<22}{'v3.13':<22}{'Δ':<15}{'v311 n':<10}{'v313 n'}")
    print("-" * 110)
    for cancer in cancers:
        m311, s311, n311 = summary.get((cancer, "v3.11"), (None, 0, 0))
        m313, s313, n313 = summary.get((cancer, "v3.13"), (None, 0, 0))
        s311_str = f"{m311:.4f}±{s311:.4f} ({n311})" if m311 is not None else "  N/A"
        s313_str = f"{m313:.4f}±{s313:.4f} ({n313})" if m313 is not None else "  N/A"
        delta = f"{m313-m311:+.4f}" if m311 is not None and m313 is not None else " N/A"
        print(f"{cancer:<10}{s311_str:<22}{s313_str:<22}{delta:<15}{n311:<10}{n313}")

if __name__ == "__main__":
    main()
