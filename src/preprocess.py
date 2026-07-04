import numpy as np
import pandas as pd
from src.config import PROC, FIG, TAB

IGP = {"lat_min": 27.5, "lat_max": 32.5, "lon_min": 73.5, "lon_max": 81.0}
HARD_BOUND = 150.0  # meters; |target| beyond this is a data error

def fix_columns(df, log):
    ren = {}
    for c in df.columns:
        cl = str(c)
        if "2m" in cl and "max" not in cl and "min" not in cl and cl != "t2m":
            ren[c] = "t2m"
    if ren:
        log.info(f"renaming columns: {ren}")
        df = df.rename(columns=ren)
    assert "t2m" in df.columns, "t2m column not resolved"
    return df

def clean_target(df, log):
    n0 = len(df)
    before = df["target"].describe().to_dict()
    df = df[df["target"].abs() <= HARD_BOUND].copy()
    n1 = len(df)
    log.info(f"hard-bound |target|<= {HARD_BOUND}m: removed {n0-n1} ({100*(n0-n1)/n0:.3f}%)")

    def mad_keep(s):
        if len(s) < 8:
            return pd.Series(True, index=s.index)
        med = s.median(); mad = (s - med).abs().median()
        if mad == 0:
            return pd.Series(True, index=s.index)
        return (s - med).abs() <= 6 * 1.4826 * mad

    keep = df.groupby("station_id")["target"].transform(mad_keep)
    n_mad = int((~keep).sum())
    df = df[keep].copy()
    log.info(f"per-well MAD (k=6) filter: removed {n_mad}")
    after = df["target"].describe().to_dict()
    log.info(f"target BEFORE: {before}")
    log.info(f"target AFTER:  {after}")
    return df, {"removed_hard_bound": n0 - n1, "removed_mad": n_mad,
                "before": before, "after": after}

def build_quarterly(df, log):
    from src.logging_utils import Timer
    with Timer(log, "quarterly resample"):
        panel = (df.set_index("datetime").groupby("station_id")
                 .resample("QS")
                 .agg(target=("target", "mean"),
                      rainfall=("rainfall", "mean"),
                      t2m=("t2m", "mean"),
                      t2m_max=("t2m_max", "mean"),
                      t2m_min=("t2m_min", "mean"),
                      lat=("latitude", "mean"),
                      lon=("longitude", "mean"),
                      wellDepth=("wellDepth", "mean"),
                      n_raw=("target", "size"))
                 .reset_index()
                 .dropna(subset=["target"]))
        panel["year"] = panel["datetime"].dt.year
    log.info(f"quarterly panel: shape={panel.shape}, wells={panel.station_id.nunique()}")
    panel.to_parquet(PROC / "quarterly_panel.parquet", index=False)
    return panel

def _grid(counts, label, log):
    g = {f"min_q>={k}": int((counts >= k).sum()) for k in (8, 12, 16, 20, 24, 32)}
    log.info(f"[{label}] quarterly cohort grid: {g}")
    return g

def run(df, log):
    out = {}
    df = fix_columns(df, log)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df, out["cleaning"] = clean_target(df, log)

    panel = build_quarterly(df, log)
    out["panel_shape"] = list(panel.shape)
    out["panel_wells"] = int(panel.station_id.nunique())

    nat_q = panel.groupby("station_id").size()
    igp = panel[panel.lat.between(IGP["lat_min"], IGP["lat_max"]) &
                panel.lon.between(IGP["lon_min"], IGP["lon_max"])]
    igp_q = igp.groupby("station_id").size()
    out["national_cohort_grid"] = _grid(nat_q, "national", log)
    out["igp_cohort_grid"] = _grid(igp_q, "IGP", log)

    oby = panel.groupby("year").size()
    out["obs_by_year"] = {int(y): int(v) for y, v in oby.items()}
    log.info(f"obs_by_year: {out['obs_by_year']}")

    TH = 16  # >=16 quarters = 4 years of history
    nat_cohort = nat_q[nat_q >= TH].index
    igp_cohort = igp_q[igp_q >= TH].index
    pd.Series(nat_cohort, name="station_id").to_csv(TAB / "cohort_national_q16.csv", index=False)
    pd.Series(igp_cohort, name="station_id").to_csv(TAB / "cohort_igp_q16.csv", index=False)
    out["cohort_national_q16"] = int(len(nat_cohort))
    out["cohort_igp_q16"] = int(len(igp_cohort))

    sub = panel[panel.station_id.isin(nat_cohort)]
    out["split_counts_national_q16"] = {
        "train_<=2018": int((sub.year <= 2018).sum()),
        "val_2019_2020": int(sub.year.between(2019, 2020).sum()),
        "test_>=2021": int((sub.year >= 2021).sum()),
    }
    log.info(f"split_counts_national_q16: {out['split_counts_national_q16']}")

    # sample cleaned IGP wells for unit/sign confirmation
    top = igp_q.sort_values(ascending=False).head(3).index.tolist()
    igp[igp.station_id.isin(top)][["station_id", "datetime", "target", "rainfall", "t2m"]] \
        .to_csv(TAB / "sample_igp_wells.csv", index=False)
    out["sample_igp_wells"] = top

    # figures
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(); df["target"].clip(-50, 50).plot(kind="hist", bins=80)
    plt.title("target cleaned (view clipped ±50m)"); plt.xlabel("target (m)")
    plt.tight_layout(); plt.savefig(FIG / "target_hist_clean.png", dpi=150); plt.close()
    plt.figure(); nat_q.plot(kind="hist", bins=50)
    plt.title("national: quarterly points per well"); plt.tight_layout()
    plt.savefig(FIG / "national_qpoints_per_well.png", dpi=150); plt.close()
    plt.figure(); igp_q.plot(kind="hist", bins=40)
    plt.title("IGP: quarterly points per well"); plt.tight_layout()
    plt.savefig(FIG / "igp_qpoints_per_well.png", dpi=150); plt.close()
    return out
