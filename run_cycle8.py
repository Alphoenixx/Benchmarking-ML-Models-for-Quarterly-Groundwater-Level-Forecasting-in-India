import os, sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ROOT, SEED, TAB, FIG, REP
from src.logging_utils import get_logger
from src.dataset import load_panel, cohort, seasonal_scale
from src import metrics as M
from src import seq as SEQ
from src import dl_models as DL
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

log = get_logger("cycle8")
HORIZONS = (1, 2, 3, 4)
MODELS = ROOT / "outputs" / "models"; MODELS.mkdir(parents=True, exist_ok=True)
KEY = ["station_id", "period", "h", "split"]
np.random.seed(SEED); torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info(f"START cycle8 | device={device} | torch={torch.__version__}")

def ids_of(c):
    return (c["station_id"].unique().tolist()
            if isinstance(c, pd.DataFrame) else list(c))

# ---------- data ----------
panel = load_panel()
log.info(f"panel {panel.shape}")
nat_ids = ids_of(cohort(panel, min_q=16, region=None))
igp_ids = set(ids_of(cohort(panel, min_q=16, region="igp")))
nat_panel = panel[panel["station_id"].isin(nat_ids)].copy()
log.info(f"national wells={len(nat_ids)} | igp wells={len(igp_ids)}")

ss = seasonal_scale(nat_panel)
scale_map = dict(ss) if not isinstance(ss, dict) else ss

# ---------- 6-model long preds (baselines + trees + chronos), cached ----------
cache = TAB / "preds_6models_long.parquet"
if cache.exists():
    log.info(f"loading cached 6-model preds: {cache}")
    six = pd.read_parquet(cache)
else:
    log.info("building 6-model preds via run_cycle7.all_model_preds_long() ...")
    t0 = time.time()
    import run_cycle7 as c7
    six = c7.all_model_preds_long()
    six.to_parquet(cache, index=False)
    log.info(f"6-model preds {six.shape} built in {time.time()-t0:.1f}s -> {cache}")
log.info("6-model row counts:\n" + six.groupby('model').size().to_string())

# ---------- DL samples ----------
ynorm = SEQ.per_well_norm_stats(nat_panel)
smu, ssd, svals = SEQ.static_norm_stats(nat_panel)
Xs, Xt, hh, yn, yrw, meta = SEQ.build_samples(
    nat_panel, nat_ids, HORIZONS, ynorm, smu, ssd, svals, scale_map)
log.info(f"DL samples total={len(Xs)} | by split:\n"
         + meta.groupby('split').size().to_string())

def pick(sp):
    m = (meta["split"] == sp).to_numpy()
    return (Xs[m], Xt[m], hh[m], yn[m], yrw[m],
            meta["mu"].to_numpy()[m], meta["sd"].to_numpy()[m])

tr, va, te = pick("train"), pick("val"), pick("test")
log.info(f"DL n: train={len(tr[0])} val={len(va[0])} test={len(te[0])}")

# ---------- train LSTM + GRU ----------
dl_long, curves, best_val = [], {}, {}
for kind in ("lstm", "gru"):
    log.info(f"===== training {kind.upper()} =====")
    t0 = time.time()
    model, bv, curve = DL.train_model(kind, tr, va, device, log, seed=SEED)
    torch.save(model.state_dict(), MODELS / f"{kind}_global.pt")
    curves[kind], best_val[kind] = curve, bv
    log.info(f"{kind} trained in {time.time()-t0:.1f}s | best_val_rmse={bv:.4f}")
    for sp, X in (("val", va), ("test", te)):
        pr = DL.predict(model, (X[0], X[1], X[2], X[5], X[6]), device)
        mm = meta[meta["split"] == sp].reset_index(drop=True)
        dl_long.append(pd.DataFrame({
            "station_id": mm["station_id"], "split": sp, "h": mm["h"],
            "period": mm["period"], "model": kind,
            "y": X[4], "yhat": pr, "scale": mm["scale"]}))
dl_long = pd.concat(dl_long, ignore_index=True)

# ---------- unify + joint common set ----------
allp = pd.concat([six, dl_long], ignore_index=True)
n_models = allp["model"].nunique()
log.info(f"models in pool = {n_models}: {sorted(allp['model'].unique())}")
cnt = allp.groupby(KEY)["model"].nunique()
common = cnt[cnt == n_models].index
allc = allp.set_index(KEY).loc[common].reset_index()
allc.to_parquet(TAB / "all_preds_long.parquet", index=False)
sizes = allc[allc.model == "seasonal_naive"].groupby(["split", "h"]).size()
log.info("JOINT common-set sizes:\n" + sizes.to_string())

# ---------- metrics ----------
nat = M.summarize(allc)
igp = M.summarize(allc[allc["station_id"].isin(igp_ids)])
nat.to_csv(TAB / "comparison_common_national_v3.csv", index=False)
igp.to_csv(TAB / "comparison_common_igp_v3.csv", index=False)

# skill (national) vs seasonal_naive
sn = nat[nat.model == "seasonal_naive"][["split", "h", "RMSE"]].rename(
    columns={"RMSE": "RMSE_sn"})
skill = nat.merge(sn, on=["split", "h"])
skill["skill_vs_seasonal_pct"] = 100 * (1 - skill["RMSE"] / skill["RMSE_sn"])
skill.to_csv(TAB / "skill_scores_national_v3.csv", index=False)

# per-well win-rate vs seasonal (national test)
te_c = allc[allc.split == "test"].copy()
pw = (te_c.assign(ae=(te_c.y - te_c.yhat) ** 2)
      .groupby(["model", "h", "station_id"])["ae"]
      .apply(lambda s: np.sqrt(s.mean())).reset_index(name="rmse"))
base = pw[pw.model == "seasonal_naive"][["h", "station_id", "rmse"]].rename(
    columns={"rmse": "rmse_sn"})
pw = pw.merge(base, on=["h", "station_id"])
wr = (pw[pw.model != "seasonal_naive"]
      .assign(win=lambda d: (d.rmse < d.rmse_sn).astype(float))
      .groupby(["h", "model"])
      .agg(win_rate_vs_seasonal=("win", "mean"),
           n_wells=("win", "size")).reset_index())
wr.to_csv(TAB / "winrate_national_v3.csv", index=False)
log.info("WIN-RATE (national test) v3:\n" + wr.to_string(index=False))
log.info("NATIONAL v3 (test):\n" +
         nat[nat.split == "test"].to_string(index=False))
log.info("IGP v3 (test):\n" + igp[igp.split == "test"].to_string(index=False))

# ---------- figures ----------
order = ["persistence", "climatology", "seasonal_naive",
         "xgboost", "random_forest", "chronos", "lstm", "gru"]
nt = nat[nat.split == "test"]
plt.figure(figsize=(7, 5))
for mdl in order:
    d = nt[nt.model == mdl].sort_values("h")
    if len(d):
        plt.plot(d.h, d.RMSE, marker="o", label=mdl)
plt.xlabel("horizon (quarters)"); plt.ylabel("RMSE (m)")
plt.title("National test RMSE vs horizon (joint common set)")
plt.xticks([1, 2, 3, 4]); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(FIG / "rmse_vs_horizon_v3.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
piv = nt.pivot_table(index="model", columns="h", values="MASE_mean").reindex(order)
piv.plot(kind="bar", ax=plt.gca())
plt.axhline(1.0, color="k", ls="--", lw=1)
plt.ylabel("MASE (mean, per-well)"); plt.title("National test MASE by model")
plt.tight_layout(); plt.savefig(FIG / "mase_by_model_v3.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
for kind, c in curves.items():
    plt.plot(range(1, len(c) + 1), c, marker=".", label=f"{kind} val")
plt.xlabel("epoch"); plt.ylabel("val RMSE (m)"); plt.legend()
plt.title("DL training curves"); plt.tight_layout()
plt.savefig(FIG / "dl_training_curves.png", dpi=150); plt.close()

# ---------- summary ----------
summary = {
    "device": device,
    "L": SEQ.L, "MIN_CTX": SEQ.MIN_CTX,
    "dl_best_val_rmse": best_val,
    "dl_epochs": {k: len(v) for k, v in curves.items()},
    "n_models": int(n_models),
    "joint_common_sizes": {f"{s}_h{h}": int(v)
                           for (s, h), v in sizes.items()},
    "national": nat.to_dict(orient="records"),
    "igp": igp.to_dict(orient="records"),
    "skill_national": skill[["split", "h", "model", "RMSE",
                             "skill_vs_seasonal_pct"]].to_dict(orient="records"),
    "winrate_national": wr.to_dict(orient="records"),
}
with open(REP / "cycle8_summary.json", "w") as f:
    json.dump(summary, f, indent=1, default=float)
log.info("END cycle8 — wrote cycle8_summary.json")
