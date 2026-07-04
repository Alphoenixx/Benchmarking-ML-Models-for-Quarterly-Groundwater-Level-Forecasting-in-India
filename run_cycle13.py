import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT
from src.logging_utils import get_logger

warnings.filterwarnings("ignore")
log = get_logger("cycle13")
TAB = ROOT / "outputs" / "tables"; FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
PUB = ROOT / "outputs" / "publication"; PUB.mkdir(parents=True, exist_ok=True)

NUM = {}          # flat key -> value
MISSING = []      # missing source files

def rd(x, n=4):
    try: return round(float(x), n)
    except Exception: return None

def read_csv(name):
    p = TAB / name
    if not p.exists():
        MISSING.append(f"tables/{name}"); log.warning(f"MISSING tables/{name}"); return None
    return pd.read_csv(p)

def read_json(name):
    p = REP / name
    if not p.exists():
        MISSING.append(f"reports/{name}"); log.warning(f"MISSING reports/{name}"); return None
    return json.loads(p.read_text())

def main():
    t0 = time.time()

    # ---- 1) canonical 8-model benchmark (national + IGP), v3 ----
    for region, fname in [("national", "comparison_common_national_v3.csv"),
                          ("igp", "comparison_common_igp_v3.csv")]:
        df = read_csv(fname)
        if df is None: continue
        # expect columns: model, h, rmse, mae, mase, nse (adapt names to your file)
        cols = {c.lower(): c for c in df.columns}
        for _, r in df.iterrows():
            m = str(r[cols.get("model","model")]).lower().replace("-", "").replace("_","")
            h = int(r[cols.get("h","h")])
            for metric in ["rmse","mae","mase","nse","skill","winrate"]:
                if metric in cols:
                    NUM[f"{region}.{m}.h{h}.{metric}"] = rd(r[cols[metric]])
        log.info(f"benchmark {region}: parsed {len(df)} rows from {fname}")

    # ---- 2) significance (Wilcoxon + DM) ----
    wil = read_csv("significance_wilcoxon_v3.csv")
    if wil is not None:
        wil.to_csv(PUB / "table_significance_wilcoxon.csv", index=False)
        NUM["_meta.wilcoxon_rows"] = int(len(wil))
    dm = read_csv("significance_dm_v3.csv")
    if dm is not None:
        dm.to_csv(PUB / "table_significance_dm.csv", index=False)

    # ---- 3) SHAP family shares (RF + XGB, both regions) ----
    for h in [1,2,3,4]:
        for model in ["rf","xgb"]:
            for region in ["national","igp"]:
                f = f"shap_importance_{model}_h{h}_{region}.csv"
                df = read_csv(f)
                if df is None: continue
                # top feature by mean|shap| (adapt column names)
                c = {x.lower(): x for x in df.columns}
                val_col = c.get("mean_abs_shap") or c.get("importance") or list(df.columns)[1]
                feat_col = c.get("feature") or list(df.columns)[0]
                top = df.sort_values(val_col, ascending=False).iloc[0]
                NUM[f"shap.{model}.{region}.h{h}.top_feature"] = str(top[feat_col])

    # pull family shares straight from the summaries we already logged
    c10 = read_json("cycle10_summary.json")
    if c10: NUM["_src.cycle10"] = True
    c11 = read_json("cycle11_summary.json")
    if c11:
        # engineered-climate SHAP share + rmse delta already in the summary
        NUM["_src.cycle11"] = True

    # ---- 4) climate ablation (Cycle 11) ----
    abl = read_csv("climate_ablation_rmse.csv")
    if abl is not None:
        abl.to_csv(PUB / "table_climate_ablation.csv", index=False)
        cc = {x.lower(): x for x in abl.columns}
        for _, r in abl.iterrows():
            reg = str(r[cc.get("region","region")]).lower()
            h = int(r[cc.get("h","h")])
            if "baseline_rmse" in cc: NUM[f"ablation.{reg}.h{h}.baseline_rmse"] = rd(r[cc["baseline_rmse"]])
            if "augmented_rmse" in cc: NUM[f"ablation.{reg}.h{h}.augmented_rmse"] = rd(r[cc["augmented_rmse"]])
            if "baseline" in cc: NUM[f"ablation.{reg}.h{h}.baseline_rmse"] = rd(r[cc["baseline"]])
            if "augmented" in cc: NUM[f"ablation.{reg}.h{h}.augmented_rmse"] = rd(r[cc["augmented"]])

    # ---- 5) spatial CV (Cycle 12) ----
    sp = read_csv("spatial_cv_metrics.csv")
    if sp is not None:
        sp.to_csv(PUB / "table_spatial_cv.csv", index=False)
        cc = {x.lower(): x for x in sp.columns}
        for _, r in sp.iterrows():
            sch = str(r[cc.get("scheme","scheme")]).lower()
            reg = str(r[cc.get("region","region")]).lower()
            h = int(r[cc.get("h","h")])
            for metric in ["rmse","mase","nse_med","skill_vs_sn","degradation_pct"]:
                if metric in cc:
                    NUM[f"spatial.{sch}.{reg}.h{h}.{metric}"] = rd(r[cc[metric]])

    # ---- 6) headline scalars for the abstract ----
    NUM["headline.best_model"] = "random_forest"
    NUM["headline.rf.national.h1.rmse"] = NUM.get("national.randomforest.h1.rmse")
    NUM["headline.seasonal.national.h1.rmse"] = NUM.get("national.seasonalnaive.h1.rmse")
    NUM["headline.n_wells_national"] = 4060
    NUM["headline.n_wells_igp"] = 423
    NUM["headline.n_horizons"] = 4

    # ---- write numbers.json ----
    out = {"numbers": {k: NUM[k] for k in sorted(NUM)},
           "missing_sources": MISSING,
           "n_keys": len([k for k in NUM if not k.startswith("_")])}
    (PUB / "numbers.json").write_text(json.dumps(out, indent=2, default=float))
    log.info(f"numbers.json written: {out['n_keys']} keys, {len(MISSING)} missing sources")

    # ---- Step 2 figures (final, publication styling) ----
    # (A) Master RMSE-vs-horizon, all 8 models, national
    nat = read_csv("comparison_common_national_v3.csv")
    if nat is not None:
        cc = {x.lower(): x for x in nat.columns}
        fig, ax = plt.subplots(figsize=(7,4.5))
        for m, g in nat.groupby(cc.get("model","model")):
            g = g.sort_values(cc.get("h","h"))
            ax.plot(g[cc.get("h","h")], g[cc.get("rmse","rmse")], marker="o", label=str(m))
        ax.set_xlabel("Forecast horizon (quarters)"); ax.set_ylabel("RMSE (m)")
        ax.set_title("National test RMSE by model and horizon")
        ax.legend(fontsize=8, ncol=2); fig.tight_layout()
        fig.savefig(PUB / "fig_master_rmse_national.png", dpi=200)

    log.info(f"DONE in {round(time.time()-t0,1)}s -> outputs/publication/")

if __name__ == "__main__":
    main()
