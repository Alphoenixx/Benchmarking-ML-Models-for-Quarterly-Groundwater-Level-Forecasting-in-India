import json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from src.config import REP, TAB, FIG, ROOT
from src.logging_utils import get_logger, Timer
from src import dataset, features, metrics

HORIZONS = (1, 2, 3, 4)
SEED = 42
MODELS = ROOT / "outputs" / "models"; MODELS.mkdir(parents=True, exist_ok=True)

def feat_cols(df):
    drop = {"station_id"}
    for h in HORIZONS:
        drop |= {f"target_h{h}", f"target_h{h}_year"}
    return [c for c in df.columns if c not in drop]

def prep(df, h):
    sub = df[df[f"target_h{h}"].notna()].copy()
    yy = sub[f"target_h{h}_year"]
    sub["_split"] = np.where(yy <= 2018, "train",
                     np.where(yy <= 2020, "val",
                     np.where(yy <= 2023, "test", "exclude")))
    return sub[sub["_split"] != "exclude"]

def impute(train, others, cols):
    med = train[cols].median(numeric_only=True)
    def tx(fr):
        X = fr[cols].copy()
        miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
        return pd.concat([X.fillna(med), miss], axis=1)
    return tx(train), [tx(o) for o in others]

def collect(fr, ys, preds, split, h, model, scale, sink):
    for sid, yt, yh in zip(fr["station_id"].to_numpy(), ys, preds):
        sink.append((sid, split, int(h), model, float(yt), float(yh), scale.get(sid, np.nan)))

def main():
    log = get_logger("cycle5")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    panel = dataset.load_panel()
    nat = dataset.cohort(panel, min_q=16)
    scale = dataset.seasonal_scale(nat)
    igp_set = set(dataset.cohort(panel, min_q=16, region="igp")["station_id"].unique())
    log.info(f"national wells={nat.station_id.nunique()} | IGP wells={len(igp_set)}")

    recs = []
    for sid, g in tqdm(nat.groupby("station_id"), desc="features"):
        recs.extend(features.build_well(g, HORIZONS))
    df = pd.DataFrame(recs)
    log.info(f"feature table: {df.shape}")
    cols = feat_cols(df)

    rows, importances = [], {}
    for h in HORIZONS:
        sub = prep(df, h)
        tr = sub[sub._split == "train"]; va = sub[sub._split == "val"]; te = sub[sub._split == "test"]
        log.info(f"h={h}: train={len(tr)} val={len(va)} test={len(te)}")
        ytr = tr[f"target_h{h}"].to_numpy(); yva = va[f"target_h{h}"].to_numpy(); yte = te[f"target_h{h}"].to_numpy()
        Xtr, (Xva, Xte) = impute(tr, [va, te], cols)

        with Timer(log, f"xgboost h{h}"):
            m = xgb.XGBRegressor(n_estimators=700, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 random_state=SEED, n_jobs=-1, early_stopping_rounds=50)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            log.info(f"  xgb best_iteration={getattr(m,'best_iteration',None)}")
            collect(va, yva, m.predict(Xva), "val", h, "xgboost", scale, rows)
            collect(te, yte, m.predict(Xte), "test", h, "xgboost", scale, rows)
            imp = pd.Series(m.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
            importances[f"xgboost_h{h}"] = imp.head(15).round(4).to_dict()
            m.save_model(str(MODELS / f"xgb_h{h}.json"))

        with Timer(log, f"random_forest h{h}"):
            rf = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=5,
                                       random_state=SEED, n_jobs=-1)
            rf.fit(Xtr, ytr)
            collect(va, yva, rf.predict(Xva), "val", h, "random_forest", scale, rows)
            collect(te, yte, rf.predict(Xte), "test", h, "random_forest", scale, rows)

    ml = pd.DataFrame(rows, columns=["station_id", "split", "h", "model", "y", "yhat", "scale"])
    nat_m = metrics.summarize(ml)
    igp_m = metrics.summarize(ml[ml.station_id.isin(igp_set)])
    nat_m.to_csv(TAB / "ml_metrics_national.csv", index=False)
    igp_m.to_csv(TAB / "ml_metrics_igp.csv", index=False)
    log.info(f"national ML metrics:\n{nat_m.to_string(index=False)}")
    log.info(f"IGP ML metrics:\n{igp_m.to_string(index=False)}")

    comp_nat = pd.concat([pd.read_csv(TAB / "baseline_metrics_national.csv"), nat_m], ignore_index=True)
    comp_igp = pd.concat([pd.read_csv(TAB / "baseline_metrics_igp.csv"), igp_m], ignore_index=True)
    comp_nat.to_csv(TAB / "comparison_national.csv", index=False)
    comp_igp.to_csv(TAB / "comparison_igp.csv", index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = comp_nat[comp_nat.split == "test"]
    plt.figure()
    for mdl, gg in t.groupby("model"):
        plt.plot(gg["h"], gg["RMSE"], marker="o", label=mdl)
    plt.xlabel("horizon (quarters)"); plt.ylabel("RMSE (m)")
    plt.title("National test: RMSE vs horizon (all models)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "model_rmse_vs_horizon.png", dpi=150); plt.close()

    out = {"national": nat_m.to_dict("records"),
           "igp": igp_m.to_dict("records"),
           "importances": importances}
    with open(REP / "cycle5_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"WROTE cycle5_summary.json:\n{json.dumps(out, indent=2, default=str)}")
    log.info("CYCLE 5 COMPLETE")

if __name__ == "__main__":
    main()
