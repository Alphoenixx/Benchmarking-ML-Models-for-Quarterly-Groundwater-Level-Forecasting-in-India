import os, json, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, shap
from xgboost import XGBRegressor

from src.config import ROOT, SEED
from src.logging_utils import get_logger
from src import features as F           # reuse EXACT training feature builder
from src import dataset as D            # reuse cohort / split / IGP bbox
from src import interpret as I

warnings.filterwarnings("ignore")
np.random.seed(SEED)
log = get_logger("cycle10")

TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
REP = ROOT / "outputs" / "reports"
MOD = ROOT / "outputs" / "models"
for p in (TAB, FIG, REP): p.mkdir(parents=True, exist_ok=True)

HORIZONS = (1, 2, 3, 4)
N_SHAP = int(os.environ.get("N_SHAP", "4000"))   # rows per (model,h,region) subsample
LAT = (27.5, 32.5); LON = (73.5, 81.0)           # IGP bbox (same as config)
t0 = time.time()

# ---- 1. Build the SAME feature frame used in Cycle 5 training -----------------
panel = D.load_panel()
log.info(f"panel rows={len(panel)} wells={panel['station_id'].nunique()}")

# cohort = national >=16q (training cohort). Reuse the project's cohort() helper.
coh = D.cohort(panel, min_q=16, region=None)
log.info(f"national cohort rows={len(coh)} wells={coh['station_id'].nunique()}")

# Build supervised feature rows per well for all horizons, exactly as training.
recs = []
for sid, g in coh.groupby("station_id", sort=False):
    recs.extend(F.build_well(g.copy(), HORIZONS))
df = pd.DataFrame(recs)
log.info(f"feature table rows={len(df)} cols={df.shape[1]}")

def get_h_test(df, h):
    sub = df[df[f"target_h{h}"].notna()].copy()
    yy = sub[f"target_h{h}_year"]
    sub["split"] = np.where(yy <= 2018, "train",
                   np.where(yy <= 2020, "val",
                   np.where(yy <= 2023, "test", "exclude")))
    sub = sub[sub["split"] != "exclude"]
    tr = sub[sub["split"] == "train"]
    te = sub[sub["split"] == "test"].copy()
    
    drop = {"station_id"}
    for hz in HORIZONS:
        drop |= {f"target_h{hz}", f"target_h{hz}_year", f"target_h{hz}_period", "split"}
    cols = [c for c in df.columns if c not in drop]
    
    med = tr[cols].median(numeric_only=True)
    def tx(fr):
        X = fr[cols].copy()
        miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
        return pd.concat([X.fillna(med), miss], axis=1)
    
    Xte = tx(te)
    Xte["station_id"] = te["station_id"]
    latc = "lat" if "lat" in Xte.columns else "latitude"
    lonc = "lon" if "lon" in Xte.columns else "longitude"
    Xte["is_igp"] = (Xte[latc].between(*LAT)) & (Xte[lonc].between(*LON))
    return Xte


def load_model(kind, h):
    if kind == "rf":
        return joblib.load(MOD / f"rf_h{h}.joblib")
    m = XGBRegressor()
    m.load_model(str(MOD / f"xgb_h{h}.json"))
    return m

summary = {"n_shap_target": N_SHAP, "horizons": list(HORIZONS),
           "per_model": [], "elapsed_sec": None}

def subsample(df, feats, n):
    df = df.dropna(subset=feats)
    if len(df) > n:
        df = df.sample(n=n, random_state=SEED)
    return df

for kind in ("rf", "xgb"):
    for h in HORIZONS:
        test_h = get_h_test(df, h)
        log.info(f"test_h{h} rows={len(test_h)} igp_test={int(test_h['is_igp'].sum())}")
        model = load_model(kind, h)
        feats = I.model_feature_names(model, kind)
        # built-in importance
        if kind == "rf":
            builtin = np.asarray(model.feature_importances_)
        else:
            gain = model.get_booster().get_score(importance_type="gain")
            builtin = np.array([gain.get(f, 0.0) for f in feats])
        explainer = shap.TreeExplainer(model)
        for region, mask in (("national", pd.Series(True, index=test_h.index)),
                             ("igp", test_h["is_igp"])):
            sub = subsample(test_h[mask], feats, N_SHAP)
            if len(sub) < 50:
                log.info(f"[skip] {kind} h{h} {region}: only {len(sub)} rows")
                continue
            X = sub[feats]
            sv = explainer.shap_values(X, check_additivity=False)
            if isinstance(sv, list):            # safety: some versions return list
                sv = sv[0]
            mabs = I.mean_abs_shap(np.asarray(sv))
            fam = I.family_shares(feats, mabs)
            order = np.argsort(-mabs)
            top10 = [{"feature": feats[i], "mean_abs_shap": float(mabs[i]),
                      "family": I.feature_family(feats[i])} for i in order[:10]]
            rho = I.spearman(mabs, builtin)
            rec = {"model": kind, "h": h, "region": region, "n_rows": int(len(sub)),
                   "family_shares": fam, "top10": top10,
                   "spearman_shap_vs_builtin": rho}
            summary["per_model"].append(rec)
            log.info(f"{kind} h{h} {region} n={len(sub)} "
                     f"AR={fam.get('autoregressive',0):.3f} cov={fam.get('covariate',0):.3f} "
                     f"seas={fam.get('seasonal',0):.3f} rho={rho:.3f} top={top10[0]['feature']}")
            # save per-feature CSV
            pd.DataFrame({"feature": feats, "mean_abs_shap": mabs,
                          "family": [I.feature_family(f) for f in feats],
                          "builtin_importance": builtin}
                         ).sort_values("mean_abs_shap", ascending=False)\
             .to_csv(TAB / f"shap_importance_{kind}_h{h}_{region}.csv", index=False)
            # figures only for RF national (representative)
            if kind == "rf" and region == "national":
                plt.figure()
                shap.summary_plot(np.asarray(sv), X, show=False, max_display=15)
                plt.tight_layout(); plt.savefig(FIG / f"shap_beeswarm_rf_h{h}.png", dpi=150); plt.close()
                for featname in ("y_lag1", "y_roll4_mean"):
                    if featname in feats:
                        plt.figure()
                        shap.dependence_plot(featname, np.asarray(sv), X,
                                             interaction_index=None, show=False)
                        plt.tight_layout()
                        plt.savefig(FIG / f"shap_dep_{featname}_rf_h{h}.png", dpi=150); plt.close()

# family-share stacked bar across horizons (RF national)
rf_nat = [r for r in summary["per_model"] if r["model"] == "rf" and r["region"] == "national"]
if rf_nat:
    fams = ["autoregressive", "seasonal", "covariate", "static", "missing_flag", "other"]
    data = {fam: [next((r["family_shares"].get(fam, 0.0) for r in rf_nat if r["h"] == h), 0.0)
                  for h in HORIZONS] for fam in fams}
    plt.figure(figsize=(7, 4)); bottom = np.zeros(len(HORIZONS))
    for fam in fams:
        plt.bar([str(h) for h in HORIZONS], data[fam], bottom=bottom, label=fam)
        bottom += np.array(data[fam])
    plt.ylabel("SHAP importance share"); plt.xlabel("horizon (quarters)")
    plt.title("RF national — feature-family attribution"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG / "shap_family_shares_rf_national.png", dpi=150); plt.close()

summary["elapsed_sec"] = round(time.time() - t0, 1)
(REP / "cycle10_summary.json").write_text(json.dumps(summary, indent=2))
log.info(f"DONE in {summary['elapsed_sec']}s -> outputs/reports/cycle10_summary.json")
