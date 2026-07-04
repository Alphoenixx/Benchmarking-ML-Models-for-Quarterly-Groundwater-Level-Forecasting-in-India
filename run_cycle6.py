import json, platform
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from src.config import REP, TAB, FIG, ROOT
from src.logging_utils import get_logger, Timer
from src import dataset, features, baselines, metrics

HORIZONS = (1, 2, 3, 4)
SEED = 42
NAIVE = "seasonal_naive"
ALL_MODELS = ["persistence", "seasonal_naive", "climatology", "xgboost", "random_forest"]
MODELS = ROOT / "outputs" / "models"; MODELS.mkdir(parents=True, exist_ok=True)

def feat_cols(df):
    return [c for c in df.columns if c != "station_id" and not c.startswith("target_h")]

def impute(train, others, cols):
    med = train[cols].median(numeric_only=True)
    def tx(fr):
        X = fr[cols].copy()
        miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
        return pd.concat([X.fillna(med), miss], axis=1)
    return tx(train), [tx(o) for o in others]

def main():
    log = get_logger("cycle6")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    panel = dataset.load_panel()
    nat = dataset.cohort(panel, min_q=16)
    scale = dataset.seasonal_scale(nat)
    igp_set = set(dataset.cohort(panel, min_q=16, region="igp")["station_id"].unique())
    log.info(f"national wells={nat.station_id.nunique()} | IGP wells={len(igp_set)}")

    # --- baseline predictions (with period) ---
    brows = []
    for sid, g in tqdm(nat.groupby("station_id"), desc="baselines"):
        brows.extend(baselines.per_well_eval(dataset.reindex_well(g), HORIZONS, include_period=True))
    bdf = pd.DataFrame(brows, columns=["station_id", "split", "h", "model", "y", "yhat", "scale", "period"])
    log.info(f"baseline preds: {bdf.shape}")

    # --- feature table + tree predictions (with period) ---
    frecs = []
    for sid, g in tqdm(nat.groupby("station_id"), desc="features"):
        frecs.extend(features.build_well(g, HORIZONS))
    fdf = pd.DataFrame(frecs)
    cols = feat_cols(fdf)

    tparts = []
    for h in HORIZONS:
        sub = fdf[fdf[f"target_h{h}"].notna()].copy()
        yy = sub[f"target_h{h}_year"]
        sub["split"] = np.where(yy <= 2018, "train",
                        np.where(yy <= 2020, "val",
                        np.where(yy <= 2023, "test", "exclude")))
        sub = sub[sub["split"] != "exclude"]
        tr = sub[sub.split == "train"]
        ev = sub[sub.split.isin(["val", "test"])].reset_index(drop=True)
        ytr = tr[f"target_h{h}"].to_numpy()
        Xtr, (Xev,) = impute(tr, [ev], cols)

        m = xgb.XGBRegressor(); m.load_model(str(MODELS / f"xgb_h{h}.json"))
        pxgb = m.predict(Xev)
        with Timer(log, f"random_forest h{h}"):
            rf = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                       random_state=SEED, n_jobs=-1)
            rf.fit(Xtr, ytr); joblib.dump(rf, MODELS / f"rf_h{h}.joblib")
        prf = rf.predict(Xev)

        base = pd.DataFrame({
            "station_id": ev["station_id"].to_numpy(), "split": ev["split"].to_numpy(),
            "h": h, "y": ev[f"target_h{h}"].astype(float).to_numpy(),
            "period": ev[f"target_h{h}_period"].to_numpy(),
            "scale": ev["station_id"].map(scale).to_numpy(),
        })
        for name, pred in [("xgboost", pxgb), ("random_forest", prf)]:
            d = base.copy(); d["model"] = name; d["yhat"] = pred.astype(float)
            tparts.append(d)
    tdf = pd.concat(tparts, ignore_index=True)
    log.info(f"tree preds: {tdf.shape}")

    # --- common set: (station, period, h) present for ALL models ---
    allpred = pd.concat([bdf, tdf], ignore_index=True)
    allpred["_k"] = list(zip(allpred.station_id, allpred.period, allpred.h))
    cnt = allpred.groupby("_k")["model"].nunique()
    common = set(cnt[cnt == len(ALL_MODELS)].index)
    com = allpred[allpred["_k"].isin(common)].drop(columns="_k").copy()
    sizes = com[com.model == NAIVE].groupby(["split", "h"]).size().to_dict()
    log.info(f"common-set sizes (seasonal_naive rows) per (split,h): {sizes}")

    # --- metrics on common set ---
    nat_c = metrics.summarize(com)
    igp_c = metrics.summarize(com[com.station_id.isin(igp_set)])
    nat_c.to_csv(TAB / "comparison_common_national.csv", index=False)
    igp_c.to_csv(TAB / "comparison_common_igp.csv", index=False)
    log.info(f"COMMON national metrics:\n{nat_c.to_string(index=False)}")
    log.info(f"COMMON IGP metrics:\n{igp_c.to_string(index=False)}")

    # --- skill score vs seasonal-naive (RMSE % reduction) ---
    def skill(mdf, split="test"):
        d = mdf[mdf.split == split]
        base = d[d.model == NAIVE].set_index("h")["RMSE"]
        rows = []
        for _, r in d.iterrows():
            b = base.get(r["h"], np.nan)
            rows.append({"split": split, "h": int(r["h"]), "model": r["model"],
                         "RMSE": r["RMSE"],
                         "skill_vs_seasonal_pct": round(100 * (1 - r["RMSE"] / b), 2) if b else None})
        return pd.DataFrame(rows)
    sk = pd.concat([skill(nat_c, "test"), skill(nat_c, "val")], ignore_index=True)
    sk.to_csv(TAB / "skill_scores_national.csv", index=False)
    log.info(f"SKILL (national) vs seasonal-naive:\n{sk.to_string(index=False)}")

    # --- per-well win-rate vs seasonal-naive (test, national) ---
    def winrate(df, split="test"):
        d = df[df.split == split].copy()
        d["ae"] = (d["yhat"] - d["y"]).abs()
        pw = d.groupby(["station_id", "h", "model"])["ae"].mean().reset_index()
        piv = pw.pivot_table(index=["station_id", "h"], columns="model", values="ae")
        rows = []
        for model in ALL_MODELS:
            if model == NAIVE:
                continue
            for h in HORIZONS:
                sl = piv.xs(h, level="h")[[model, NAIVE]].dropna()
                wr = float((sl[model] < sl[NAIVE]).mean()) if len(sl) else np.nan
                rows.append({"h": h, "model": model,
                             "win_rate_vs_seasonal": round(wr, 3), "n_wells": int(len(sl))})
        return pd.DataFrame(rows)
    wr = winrate(com, "test")
    wr.to_csv(TAB / "winrate_national.csv", index=False)
    log.info(f"WIN-RATE (national test) vs seasonal-naive:\n{wr.to_string(index=False)}")

    # --- figures ---
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = nat_c[nat_c.split == "test"]
    plt.figure()
    for mdl, gg in t.groupby("model"):
        plt.plot(gg["h"], gg["RMSE"], marker="o", label=mdl)
    plt.xlabel("horizon (quarters)"); plt.ylabel("RMSE (m)")
    plt.title("National test (common set): RMSE vs horizon"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "rmse_vs_horizon_common.png", dpi=150); plt.close()

    plt.figure()
    for mdl, gg in t.groupby("model"):
        plt.plot(gg["h"], gg["MASE_mean"], marker="o", label=mdl)
    plt.axhline(1.0, ls="--", c="k", lw=0.8)
    plt.xlabel("horizon (quarters)"); plt.ylabel("MASE (per-well mean)")
    plt.title("National test (common set): MASE by model"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "mase_by_model.png", dpi=150); plt.close()

    out = {"national": nat_c.to_dict("records"), "igp": igp_c.to_dict("records"),
           "skill_national": sk.to_dict("records"), "winrate_national": wr.to_dict("records"),
           "common_sizes": {f"{k[0]}_h{k[1]}": v for k, v in sizes.items()}}
    with open(REP / "cycle6_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"WROTE cycle6_summary.json:\n{json.dumps(out, indent=2, default=str)}")
    log.info("CYCLE 6 COMPLETE")

if __name__ == "__main__":
    main()
