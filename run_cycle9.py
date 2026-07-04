import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.config import ROOT, SEED
from src.logging_utils import get_logger
from src import stats as st

HZ = [1, 2, 3, 4]
MIN_N = 2                      # min per-well test points for per-well RMSE
LAT, LON = (27.5, 32.5), (73.5, 81.0)   # IGP bbox (matches config)

def main():
    t0 = time.time()
    log = get_logger("cycle9")
    TAB = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
    REP = ROOT / "outputs" / "reports"; PROC = ROOT / "data" / "processed"
    for p in (TAB, FIG, REP):
        p.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    log.info("START cycle9 — significance + stratified robustness")
    
    preds = pd.read_parquet(TAB / "all_preds_long.parquet")
    log.info(f"all_preds_long {preds.shape} cols={list(preds.columns)}")
    log.info(f"models in pool = {len(preds.model.unique())}: {sorted(preds.model.unique())}")
    
    panel = pd.read_parquet(PROC / "quarterly_panel.parquet")
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    geo = panel.groupby("station_id").agg(lat=("lat", "first"), lon=("lon", "first")).reset_index()
    hist = (panel[(panel.datetime.dt.year <= 2018) & panel.target.notna()]
            .groupby("station_id").size().rename("train_obs").reset_index())
    stations = geo.merge(hist, on="station_id", how="left").fillna({"train_obs": 0})
    stations["igp"] = stations.lat.between(*LAT) & stations.lon.between(*LON)
    igp_wells = set(stations.loc[stations.igp, "station_id"])
    log.info(f"stations={len(stations)}  igp_wells={len(igp_wells)}")
    
    pw = st.per_well_errors(preds)
    pw_nat = pw
    pw_igp = pw[pw.station_id.isin(igp_wells)]

    # 1) pairwise Wilcoxon + Diebold-Mariano ---------------------------------
    pairs = [("random_forest", "xgboost"), ("random_forest", "chronos"),
             ("random_forest", "gru"), ("chronos", "gru"), ("chronos", "lstm"),
             ("gru", "lstm"), ("chronos", "seasonal_naive"),
             ("gru", "seasonal_naive"), ("random_forest", "seasonal_naive")]
    wil, dm = [], []
    for region, pwr, dfr in [("national", pw_nat, preds),
                             ("igp", pw_igp, preds[preds.station_id.isin(igp_wells)])]:
        for h in HZ:
            for a, b in pairs:
                r = st.wilcoxon_pair(pwr, "test", h, a, b, "rmse", MIN_N); r["region"] = region; wil.append(r)
                q = st.diebold_mariano(dfr, "test", h, a, b, "se"); q["region"] = region; dm.append(q)
    wil = pd.DataFrame(wil); dm = pd.DataFrame(dm)
    wil.to_csv(TAB / "significance_wilcoxon_v3.csv", index=False)
    dm.to_csv(TAB / "significance_dm_v3.csv", index=False)
    log.info("wrote significance_wilcoxon_v3.csv + significance_dm_v3.csv")
    log.info("national Wilcoxon (test) vs RF:\n" +
             wil[(wil.region == "national") & (wil.model_a == "random_forest")]
             [["h", "model_b", "n_wells", "median_diff", "a_better_frac", "p_value"]].to_string(index=False))

    # 2) binomial win-rate significance vs seasonal --------------------------
    models = [m for m in sorted(preds.model.unique()) if m != "seasonal_naive"]
    bw = []
    for region, pwr in [("national", pw_nat), ("igp", pw_igp)]:
        for h in HZ:
            for m in models:
                r = st.binom_winrate(pwr, "test", h, m, "seasonal_naive", MIN_N)
                r["region"] = region; bw.append(r)
    bw = pd.DataFrame(bw); bw.to_csv(TAB / "winrate_significance_v3.csv", index=False)
    log.info("wrote winrate_significance_v3.csv")

    # 3) stratify NATIONAL by train-history tercile --------------------------
    test_wells = set(preds[preds.split == "test"].station_id.unique())
    nw = stations[(stations.train_obs > 0) & stations.station_id.isin(test_wells)].copy()
    q1, q2 = nw.train_obs.quantile([1 / 3, 2 / 3]).tolist()
    nw["hist_tercile"] = np.where(nw.train_obs <= q1, "short",
                                  np.where(nw.train_obs <= q2, "medium", "long"))
    tmap = dict(zip(nw.station_id, nw.hist_tercile))
    log.info(f"history terciles: q1={q1:.0f} q2={q2:.0f} counts={nw.hist_tercile.value_counts().to_dict()}")
    
    d = preds[preds.split == "test"].copy()
    d["hist_tercile"] = d.station_id.map(tmap)
    d = d[d.hist_tercile.notna()].copy()
    d["se"] = (d.y - d.yhat) ** 2
    strat = (d.groupby(["hist_tercile", "h", "model"], observed=True)
             .agg(rmse=("se", lambda s: float(np.sqrt(np.mean(s)))), n=("se", "size")).reset_index())
    sea = strat[strat.model == "seasonal_naive"][["hist_tercile", "h", "rmse"]].rename(columns={"rmse": "rmse_sea"})
    strat = strat.merge(sea, on=["hist_tercile", "h"])
    strat["skill_pct"] = 100 * (1 - strat.rmse / strat.rmse_sea)
    strat.to_csv(TAB / "strat_history_national_v3.csv", index=False)
    log.info("wrote strat_history_national_v3.csv")

    # 4) per-well NSE distribution (median + fraction > 0) -------------------
    dd = preds[preds.split == "test"].copy()
    dd["sse"] = (dd.y - dd.yhat) ** 2
    g = dd.groupby(["station_id", "h", "model"], observed=True)
    ag = g.agg(sse=("sse", "sum"), n=("y", "size"),
               yss=("y", lambda s: float(np.sum((s - s.mean()) ** 2)))).reset_index()
    ag["nse"] = np.where(ag.yss > 0, 1 - ag.sse / ag.yss, np.nan)
    ag["region"] = np.where(ag.station_id.isin(igp_wells), "igp", "national")
    nse = (ag.dropna(subset=["nse"]).assign(pos=lambda x: x.nse > 0)
           .groupby(["region", "h", "model"], observed=True)
           .agg(nse_med=("nse", "median"), frac_pos=("pos", "mean"), n=("nse", "size")).reset_index())
    nse.to_csv(TAB / "nse_distribution_v3.csv", index=False)
    log.info("wrote nse_distribution_v3.csv")

    # figure: skill vs history tercile ---------------------------------------
    order = ["short", "medium", "long"]
    focus = ["random_forest", "xgboost", "chronos", "gru", "lstm"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
    for i, h in enumerate(HZ):
        sub = strat[strat.h == h]
        for m in focus:
            s = sub[sub.model == m].set_index("hist_tercile").reindex(order)
            ax[i].plot(order, s.skill_pct.values, marker="o", label=m)
        ax[i].axhline(0, color="k", lw=0.8, ls="--"); ax[i].set_title(f"h={h}")
        ax[i].set_xlabel("train-history tercile")
        if i == 0:
            ax[i].set_ylabel("skill vs seasonal (% RMSE)")
    ax[-1].legend(fontsize=8)
    fig.suptitle("National skill by well training-history length (v3)")
    fig.tight_layout(); fig.savefig(FIG / "skill_vs_history_national_v3.png", dpi=150); plt.close(fig)
    log.info("wrote skill_vs_history_national_v3.png")

    summary = {
        "min_n_per_well": MIN_N, "n_stations_total": int(len(stations)),
        "n_igp_wells": int(len(igp_wells)),
        "history_tercile_cuts": {"q1_train_obs": float(q1), "q2_train_obs": float(q2)},
        "history_tercile_counts": nw.hist_tercile.value_counts().to_dict(),
        "wilcoxon": wil.to_dict(orient="records"),
        "diebold_mariano": dm.to_dict(orient="records"),
        "winrate_significance": bw.to_dict(orient="records"),
        "strat_history_national": strat.to_dict(orient="records"),
        "nse_distribution": nse.to_dict(orient="records"),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (REP / "cycle9_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info(f"END cycle9 in {summary['elapsed_sec']}s")

if __name__ == "__main__":
    main()
