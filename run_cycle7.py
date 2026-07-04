import json, platform
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm import tqdm
import xgboost as xgb
from src.config import REP, TAB, FIG, ROOT
from src.logging_utils import get_logger, Timer
from src import dataset, features, baselines, metrics

HORIZONS = (1, 2, 3, 4)
SEED = 42
NAIVE = "seasonal_naive"
BASE_MODELS = ["persistence", "seasonal_naive", "climatology", "xgboost", "random_forest"]
ALL_MODELS = BASE_MODELS + ["chronos"]
MODELS_DIR = ROOT / "outputs" / "models"
MIN_CTX = 4          # minimum observed points before origin for Chronos
CH_BATCH = 256       # Chronos inference batch size

def feat_cols(df):
    return [c for c in df.columns if c != "station_id" and not c.startswith("target_h")]

def impute(train, others, cols):
    med = train[cols].median(numeric_only=True)
    def tx(fr):
        X = fr[cols].copy()
        miss = X.isna().astype(int); miss.columns = [f"{c}__miss" for c in cols]
        return pd.concat([X.fillna(med), miss], axis=1)
    return tx(train), [tx(o) for o in others]

# ---------- baselines + trees (LOAD saved models, no retrain) ----------
def base_and_tree_preds(nat, scale, log):
    brows = []
    for _, g in tqdm(nat.groupby("station_id"), desc="baselines"):
        brows.extend(baselines.per_well_eval(dataset.reindex_well(g), HORIZONS, include_period=True))
    bdf = pd.DataFrame(brows, columns=["station_id", "split", "h", "model", "y", "yhat", "scale", "period"])

    frecs = []
    for _, g in tqdm(nat.groupby("station_id"), desc="features"):
        frecs.extend(features.build_well(g, HORIZONS))
    fdf = pd.DataFrame(frecs)
    cols = feat_cols(fdf)

    tparts = []
    for h in HORIZONS:
        sub = fdf[fdf[f"target_h{h}"].notna()].copy()
        yy = sub[f"target_h{h}_year"]
        sub["split"] = np.where(yy <= 2018, "train", np.where(yy <= 2020, "val",
                        np.where(yy <= 2023, "test", "exclude")))
        sub = sub[sub["split"] != "exclude"]
        tr = sub[sub.split == "train"]
        ev = sub[sub.split.isin(["val", "test"])].reset_index(drop=True)
        _, (Xev,) = impute(tr, [ev], cols)

        xgb_m = xgb.XGBRegressor(); xgb_m.load_model(str(MODELS_DIR / f"xgb_h{h}.json"))
        rf_m = joblib.load(MODELS_DIR / f"rf_h{h}.joblib")
        pred = {"xgboost": xgb_m.predict(Xev), "random_forest": rf_m.predict(Xev)}

        base = pd.DataFrame({
            "station_id": ev["station_id"].to_numpy(), "split": ev["split"].to_numpy(),
            "h": h, "y": ev[f"target_h{h}"].astype(float).to_numpy(),
            "period": ev[f"target_h{h}_period"].to_numpy(),
            "scale": ev["station_id"].map(scale).to_numpy(),
        })
        for name, p in pred.items():
            d = base.copy(); d["model"] = name; d["yhat"] = np.asarray(p, dtype=float)
            tparts.append(d)
    tdf = pd.concat(tparts, ignore_index=True)
    log.info(f"baseline preds {bdf.shape} | tree preds {tdf.shape}")
    return pd.concat([bdf, tdf], ignore_index=True)

# ---------- Chronos ----------
def load_chronos(log):
    from chronos import BaseChronosPipeline
    for name in ["amazon/chronos-2", "amazon/chronos-bolt-base"]:
        try:
            try:
                pipe = BaseChronosPipeline.from_pretrained(name, device_map="cpu", torch_dtype=torch.float32)
            except TypeError:
                pipe = BaseChronosPipeline.from_pretrained(name)
            log.info(f"Loaded Chronos model: {name}")
            return pipe, name
        except Exception as e:
            log.info(f"Could not load {name}: {repr(e)[:200]}")
    raise RuntimeError("No Chronos model could be loaded")

def forecast_batch(pipe, batch, H):
    """Return [b, H] point (median) forecasts, robust to API shape."""
    try:
        q, _ = pipe.predict_quantiles(batch, prediction_length=H, quantile_levels=[0.5])
        if isinstance(q, (list, tuple)):
            q = torch.stack(q)
        arr = q.cpu().numpy()
        if arr.ndim == 3:
            return arr[:, :, 0]
        return arr
    except Exception:
        out = pipe.predict(batch, prediction_length=H)
        if isinstance(out, (list, tuple)):
            out = torch.stack(out)
        arr = out.cpu().numpy()
        if arr.ndim == 3:
            return np.median(arr, axis=1)
        return arr

def chronos_preds(nat, com_keys, pipe, log):
    """com_keys: DataFrame with station_id, period, h, split, y, scale (from seasonal_naive rows)."""
    uni_ctx, uni_key = [], []          # unique (sid,o) contexts
    seen = set()
    meta = []                          # (sid,split,h,y,scale,period,o)
    for sid, g in tqdm(nat.groupby("station_id"), desc="chronos-ctx"):
        gg = dataset.reindex_well(g).reset_index(drop=True)
        dates = [str(pd.Timestamp(t).date()) for t in gg["datetime"]]
        pos = {d: i for i, d in enumerate(dates)}
        y = gg["target"].to_numpy(dtype=float)
        sub = com_keys[com_keys.station_id == sid]
        if sub.empty:
            continue
        for _, r in sub.iterrows():
            j = pos.get(str(r["period"]))
            if j is None:
                continue
            o = j - int(r["h"])
            if o < 0:
                continue
            ctx_raw = y[:o + 1]
            if np.isfinite(ctx_raw).sum() < MIN_CTX:
                continue
            arr = pd.Series(ctx_raw).interpolate(limit_direction="both").to_numpy()
            if not np.isfinite(arr).all():
                continue
            key = (sid, o)
            if key not in seen:
                seen.add(key); uni_ctx.append(torch.tensor(arr, dtype=torch.float32)); uni_key.append(key)
            meta.append((sid, r["split"], int(r["h"]), float(r["y"]), float(r["scale"]), str(r["period"]), o))

    log.info(f"Chronos: {len(uni_ctx)} unique contexts, {len(meta)} target rows")
    preds = {}
    with Timer(log, "chronos inference"):
        for i in tqdm(range(0, len(uni_ctx), CH_BATCH), desc="chronos-predict"):
            batch = uni_ctx[i:i + CH_BATCH]
            med = forecast_batch(pipe, batch, max(HORIZONS))   # [b, 4]
            for k, key in enumerate(uni_key[i:i + CH_BATCH]):
                preds[key] = np.asarray(med[k], dtype=float)

    rows = []
    for sid, split, h, yt, scale, period, o in meta:
        p = preds.get((sid, o))
        if p is None:
            continue
        p = np.asarray(p).flatten()
        if len(p) < h:
            continue
        rows.append((sid, split, h, "chronos", yt, float(p[h - 1]), scale, period))
    return pd.DataFrame(rows, columns=["station_id", "split", "h", "model", "y", "yhat", "scale", "period"])

# ---------- metrics helpers (same as Cycle 6) ----------
def skill(mdf, split):
    d = mdf[mdf.split == split]
    base = d[d.model == NAIVE].set_index("h")["RMSE"]
    out = []
    for _, r in d.iterrows():
        b = base.get(r["h"], np.nan)
        out.append({"split": split, "h": int(r["h"]), "model": r["model"], "RMSE": r["RMSE"],
                    "skill_vs_seasonal_pct": round(100 * (1 - r["RMSE"] / b), 2) if b else None})
    return pd.DataFrame(out)

def winrate(df, split="test"):
    d = df[df.split == split].copy()
    d["ae"] = (d["yhat"] - d["y"]).abs()
    pw = d.groupby(["station_id", "h", "model"])["ae"].mean().reset_index()
    piv = pw.pivot_table(index=["station_id", "h"], columns="model", values="ae")
    out = []
    for model in ALL_MODELS:
        if model == NAIVE:
            continue
        for h in HORIZONS:
            sl = piv.xs(h, level="h")[[model, NAIVE]].dropna()
            wr = float((sl[model] < sl[NAIVE]).mean()) if len(sl) else np.nan
            out.append({"h": h, "model": model, "win_rate_vs_seasonal": round(wr, 3), "n_wells": int(len(sl))})
    return pd.DataFrame(out)

def all_model_preds_long():
    """Long-format predictions for baselines + RF + XGB + Chronos.
    Columns: station_id, split, h, period, model, y, yhat, scale
    Reuses the exact Cycle 7 builders (no re-derivation)."""
    log = get_logger("cycle7")
    panel = dataset.load_panel()
    nat = dataset.cohort(panel, min_q=16)
    scale = dataset.seasonal_scale(nat)

    base_df = base_and_tree_preds(nat, scale, log)
    base_df["_k"] = list(zip(base_df.station_id, base_df.period, base_df.h))
    cnt = base_df.groupby("_k")["model"].nunique()
    common5 = set(cnt[cnt == len(BASE_MODELS)].index)
    com_keys = base_df[(base_df.model == NAIVE) & (base_df["_k"].isin(common5))][
        ["station_id", "period", "h", "split", "y", "scale"]].copy()

    pipe, _ = load_chronos(log)
    chronos_df = chronos_preds(nat, com_keys, pipe, log)

    if "_k" in base_df.columns:
        base_df = base_df.drop(columns=["_k"])
    if "_k" in chronos_df.columns:
        chronos_df = chronos_df.drop(columns=["_k"])

    out = pd.concat([base_df, chronos_df], ignore_index=True)
    need = ["station_id", "split", "h", "period", "model", "y", "yhat", "scale"]
    missing = [c for c in need if c not in out.columns]
    assert not missing, f"all_model_preds_long missing cols: {missing}"
    return out[need]

def main():
    log = get_logger("cycle7")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()} | torch {torch.__version__}")
    panel = dataset.load_panel()
    nat = dataset.cohort(panel, min_q=16)
    scale = dataset.seasonal_scale(nat)
    igp_set = set(dataset.cohort(panel, min_q=16, region="igp")["station_id"].unique())

    allbase = base_and_tree_preds(nat, scale, log)
    allbase["_k"] = list(zip(allbase.station_id, allbase.period, allbase.h))
    cnt = allbase.groupby("_k")["model"].nunique()
    common5 = set(cnt[cnt == len(BASE_MODELS)].index)
    com_keys = allbase[(allbase.model == NAIVE) & (allbase["_k"].isin(common5))][
        ["station_id", "period", "h", "split", "y", "scale"]].copy()
    log.info(f"common5 keys: {len(common5)}")

    pipe, ch_name = load_chronos(log)
    chdf = chronos_preds(nat, com_keys, pipe, log)
    chdf["_k"] = list(zip(chdf.station_id, chdf.period, chdf.h))
    common6 = common5 & set(chdf["_k"])
    log.info(f"common6 keys: {len(common6)} (dropped {len(common5) - len(common6)} for short Chronos context)")

    full = pd.concat([allbase, chdf], ignore_index=True)
    com = full[full["_k"].isin(common6)].drop(columns="_k").copy()
    sizes = com[com.model == NAIVE].groupby(["split", "h"]).size().to_dict()
    log.info(f"common6 sizes per (split,h): {sizes}")

    nat_c = metrics.summarize(com)
    igp_c = metrics.summarize(com[com.station_id.isin(igp_set)])
    nat_c.to_csv(TAB / "comparison_common_national_v2.csv", index=False)
    igp_c.to_csv(TAB / "comparison_common_igp_v2.csv", index=False)
    log.info(f"COMMON6 national:\n{nat_c.to_string(index=False)}")
    log.info(f"COMMON6 IGP:\n{igp_c.to_string(index=False)}")

    sk = pd.concat([skill(nat_c, "test"), skill(nat_c, "val")], ignore_index=True)
    sk.to_csv(TAB / "skill_scores_national_v2.csv", index=False)
    log.info(f"SKILL (national):\n{sk.to_string(index=False)}")

    wr = winrate(com, "test")
    wr.to_csv(TAB / "winrate_national_v2.csv", index=False)
    log.info(f"WIN-RATE (national test):\n{wr.to_string(index=False)}")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = nat_c[nat_c.split == "test"]
    for metric, fname in [("RMSE", "rmse_vs_horizon_v2.png"), ("MASE_mean", "mase_by_model_v2.png")]:
        plt.figure()
        for mdl, gg in t.groupby("model"):
            plt.plot(gg["h"], gg[metric], marker="o", label=mdl)
        if metric == "MASE_mean":
            plt.axhline(1.0, ls="--", c="k", lw=0.8)
        plt.xlabel("horizon (quarters)"); plt.ylabel(metric)
        plt.title(f"National test (common6, 6 models): {metric}"); plt.legend(); plt.tight_layout()
        plt.savefig(FIG / fname, dpi=150); plt.close()

    out = {"chronos_model": ch_name,
           "common5": len(common5), "common6": len(common6),
           "common6_sizes": {f"{k[0]}_h{k[1]}": v for k, v in sizes.items()},
           "national": nat_c.to_dict("records"), "igp": igp_c.to_dict("records"),
           "skill_national": sk.to_dict("records"), "winrate_national": wr.to_dict("records")}
    with open(REP / "cycle7_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"WROTE cycle7_summary.json:\n{json.dumps(out, indent=2, default=str)}")
    log.info("CYCLE 7 COMPLETE")

if __name__ == "__main__":
    main()
