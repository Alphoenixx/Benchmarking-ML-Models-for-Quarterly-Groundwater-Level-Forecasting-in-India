import json, time, warnings
import numpy as np, pandas as pd
from pathlib import Path
import joblib
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT, SEED
from src.logging_utils import get_logger
from src import dataset as D, features as F

warnings.filterwarnings("ignore")
log = get_logger("cycle12")
HORIZONS = [1, 2, 3, 4]
K = 5
TAB = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"; MOD = ROOT / "outputs" / "models"
for p in (TAB, FIG, REP): p.mkdir(parents=True, exist_ok=True)

IGP_LAT, IGP_LON = (27.5, 32.5), (73.5, 81.0)
def is_igp(lat, lon):
    return (lat >= IGP_LAT[0]) & (lat <= IGP_LAT[1]) & (lon >= IGP_LON[0]) & (lon <= IGP_LON[1])

CANON_NAT = {1: 2.2715, 2: 2.4738, 3: 2.6086, 4: 2.7028}
CANON_IGP = {1: 1.6613, 2: 1.9260, 3: 1.8582, 4: 2.1492}

def metrics(df):
    err = df["yhat"] - df["y"]
    rmse = float(np.sqrt(np.mean(err**2)))
    mase = float(np.mean(np.abs(err) / df["scale"].replace(0, np.nan)))
    nse = []
    for _, g in df.groupby("station_id"):
        denom = np.sum((g["y"] - g["y"].mean())**2)
        if denom > 0:
            nse.append(1 - np.sum((g["yhat"] - g["y"])**2) / denom)
    return rmse, mase, (float(np.median(nse)) if nse else np.nan)

def seasonal_naive_rmse(df):
    err = df["yhat_sn"] - df["y"]
    return float(np.sqrt(np.mean(err**2)))

def get_h_data(df, h):
    sub = df[df[f"target_h{h}"].notna()].copy()
    yy = sub[f"target_h{h}_year"]
    sub["split"] = np.where(yy <= 2018, "train",
                   np.where(yy <= 2020, "val",
                   np.where(yy <= 2023, "test", "exclude")))
    sub = sub[sub["split"] != "exclude"].copy()
    sub["h"] = h
    sub["period"] = sub[f"target_h{h}_period"]
    sub["y"] = sub[f"target_h{h}"]
    
    tr_mask = sub["split"] == "train"
    tr = sub[tr_mask]
    
    drop = {"station_id", "split", "h", "period", "y", "is_igp", "scale", "yhat_sn"}
    for hz in HORIZONS:
        drop |= {f"target_h{hz}", f"target_h{hz}_year", f"target_h{hz}_period"}
    cols = [c for c in df.columns if c not in drop]
    
    med = tr[cols].median(numeric_only=True)
    def tx(fr):
        X = fr[cols].copy()
        miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
        out = pd.concat([X.fillna(med), miss], axis=1)
        for c in ["station_id", "split", "h", "period", "y", "is_igp", "scale", "yhat_sn", "spatial_block", "random_block"]:
            if c in fr.columns:
                out[c] = fr[c]
        return out
    
    return tx(sub)

def main():
    t0 = time.time()
    np.random.seed(SEED)

    panel = D.load_panel()
    coh = D.cohort(panel, min_q=16, region=None)
    log.info(f"cohort rows={len(coh)} wells={coh['station_id'].nunique()}")

    recs = []
    for sid, g in coh.groupby("station_id", sort=False):
        recs.extend(F.build_well(g.copy(), HORIZONS))
    frame = pd.DataFrame(recs)
    
    scale_map = D.seasonal_scale(panel)
    frame["scale"] = frame["station_id"].map(scale_map).fillna(pd.Series(scale_map).median())
    frame["is_igp"] = is_igp(frame["lat"], frame["lon"])
    
    allp = pd.read_parquet(TAB / "all_preds_long.parquet")
    sn = allp[allp["model"] == "seasonal_naive"][["station_id", "period", "h", "yhat"]] \
            .rename(columns={"yhat": "yhat_sn"})

    wells = frame.groupby("station_id").agg(lat=("lat", "first"), lon=("lon", "first"),
                                            is_igp=("is_igp", "first")).reset_index()
    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    wells["spatial_block"] = km.fit_predict(wells[["lat", "lon"]].to_numpy())
    rng = np.random.default_rng(SEED)
    wells["random_block"] = rng.integers(0, K, size=len(wells))
    log.info("spatial block sizes:\n" + wells["spatial_block"].value_counts().sort_index().to_string())
    log.info("cluster centroids (lat,lon):\n" +
             pd.DataFrame(km.cluster_centers_, columns=["lat", "lon"]).round(2).to_string())
    log.info(f"IGP wells in cohort = {int(wells['is_igp'].sum())}")

    block_map = {"spatial": dict(zip(wells.station_id, wells.spatial_block)),
                 "random":  dict(zip(wells.station_id, wells.random_block))}
    frame["spatial_block"] = frame["station_id"].map(block_map["spatial"])
    frame["random_block"]  = frame["station_id"].map(block_map["random"])

    rows = []
    def rf_for(h):
        p = joblib.load(MOD / f"rf_h{h}.joblib").get_params()
        p["random_state"] = SEED
        p["n_jobs"] = -1
        return RandomForestRegressor(**p), list(joblib.load(MOD / f"rf_h{h}.joblib").feature_names_in_)

    nan_counts = {}
    
    for scheme, col in [("spatial", "spatial_block"), ("random", "random_block")]:
        for h in HORIZONS:
            model_tmpl, feats = rf_for(h)
            h_frame = get_h_data(frame, h)
            h_frame = h_frame.merge(sn[sn["h"] == h], on=["station_id", "period", "h"], how="left")
            nan_c = h_frame[h_frame["split"] == "test"]["yhat_sn"].isna().sum()
            nan_counts[f"{scheme}_h{h}"] = int(nan_c)
            
            pooled = []
            for k in range(K):
                tr = h_frame[(h_frame[col] != k) & (h_frame["split"] == "train")]
                te = h_frame[(h_frame[col] == k) & (h_frame["split"] == "test")]
                if len(te) == 0:
                    continue
                m = RandomForestRegressor(**model_tmpl.get_params())
                m.fit(tr[feats].to_numpy(), tr["y"].to_numpy())
                te = te.copy(); te["yhat"] = m.predict(te[feats].to_numpy())
                pooled.append(te)
                
                # Spatial integrity check
                tr_wells = set(tr['station_id'])
                te_wells = set(te['station_id'])
                overlap = tr_wells.intersection(te_wells)
                if overlap: log.warning(f"LEAKAGE! {len(overlap)} wells in both train and test fold {k}")
                
                log.info(f"{scheme} h{h} fold{k}: train_wells={len(tr_wells)} "
                         f"test_wells={len(te_wells)} test_rows={len(te)}")
            pooled = pd.concat(pooled, ignore_index=True)
            for reg, mask in [("national", np.ones(len(pooled), bool)),
                              ("igp", pooled["is_igp"].to_numpy())]:
                d = pooled[mask]
                if len(d) < 30:
                    continue
                rmse, mase, nse = metrics(d)
                sn_rmse = seasonal_naive_rmse(d.dropna(subset=["yhat_sn"]))
                skill = 100 * (1 - rmse / sn_rmse) if sn_rmse else np.nan
                canon = (CANON_NAT if reg == "national" else CANON_IGP)[h]
                degr = 100 * (rmse - canon) / canon
                rows.append(dict(scheme=scheme, region=reg, h=h, n=len(d),
                                 rmse=rmse, mase=mase, nse_med=nse, skill_vs_sn=skill,
                                 canonical_rmse=canon, degradation_pct=degr))
                log.info(f"{scheme} {reg} h{h} n={len(d)} RMSE={rmse:.4f} "
                         f"(canon {canon:.4f}, {degr:+.1f}%) MASE={mase:.4f} "
                         f"NSE={nse:.4f} skill={skill:+.1f}%")

    for h in HORIZONS:
        model_tmpl, feats = rf_for(h)
        h_frame = get_h_data(frame, h)
        h_frame = h_frame.merge(sn[sn["h"] == h], on=["station_id", "period", "h"], how="left")
        
        tr = h_frame[(~h_frame["is_igp"]) & (h_frame["split"] == "train")]
        te = h_frame[(h_frame["is_igp"]) & (h_frame["split"] == "test")].copy()
        m = RandomForestRegressor(**model_tmpl.get_params())
        m.fit(tr[feats].to_numpy(), tr["y"].to_numpy())
        te["yhat"] = m.predict(te[feats].to_numpy())
        rmse, mase, nse = metrics(te)
        sn_rmse = seasonal_naive_rmse(te.dropna(subset=["yhat_sn"]))
        skill = 100 * (1 - rmse / sn_rmse) if sn_rmse else np.nan
        canon = CANON_IGP[h]; degr = 100 * (rmse - canon) / canon
        rows.append(dict(scheme="leave_igp_out", region="igp", h=h, n=len(te),
                         rmse=rmse, mase=mase, nse_med=nse, skill_vs_sn=skill,
                         canonical_rmse=canon, degradation_pct=degr))
        log.info(f"leave_igp_out igp h{h} n={len(te)} train_wells={tr['station_id'].nunique()} "
                 f"RMSE={rmse:.4f} (canon {canon:.4f}, {degr:+.1f}%) skill={skill:+.1f}%")

    res = pd.DataFrame(rows)
    res.to_csv(TAB / "spatial_cv_metrics.csv", index=False)
    piv = res[res.region == "national"].pivot_table(index="h", columns="scheme", values="rmse")
    piv.to_csv(TAB / "spatial_cv_summary_national.csv")
    wells.to_csv(TAB / "spatial_blocks_assignment.csv", index=False)
    
    log.info(f"Seasonal Naive NaNs in test: {nan_counts}")

    summary = dict(K=K, horizons=HORIZONS, n_wells=int(len(wells)),
                   n_igp_wells=int(wells["is_igp"].sum()),
                   spatial_block_sizes=wells["spatial_block"].value_counts().sort_index().to_dict(),
                   metrics=res.to_dict("records"), elapsed_sec=round(time.time() - t0, 1))
    (REP / "cycle12_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info(f"DONE in {summary['elapsed_sec']}s -> outputs/reports/cycle12_summary.json")
    
    # FIGURES
    nat = res[res.region=="national"]
    fig, ax = plt.subplots(figsize=(6,4))
    for scheme in ["spatial","random"]:
        d = nat[nat.scheme==scheme].sort_values("h")
        ax.plot(d.h, d.rmse, marker="o", label=f"{scheme} CV")
    ax.plot(sorted(nat.h.unique()), [nat[nat.scheme=='spatial'].sort_values('h').canonical_rmse.iloc[i] for i in range(nat.h.nunique())],
            marker="s", ls="--", color="k", label="temporal (canonical)")
    ax.set_xlabel("Horizon (quarters)"); ax.set_ylabel("RMSE (m)")
    ax.set_title("RF spatial vs temporal generalization (national)"); ax.legend()
    fig.tight_layout(); fig.savefig(ROOT/"outputs"/"figures"/"rmse_spatial_vs_temporal.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6,6))
    sc = ax.scatter(wells.lon, wells.lat, c=wells.spatial_block, cmap="tab10", s=8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_title("KMeans spatial blocks (K=5)")
    fig.tight_layout(); fig.savefig(ROOT/"outputs"/"figures"/"spatial_blocks_map.png", dpi=150)
    log.info("figures saved")

if __name__ == "__main__":
    main()
