import json, platform
import pandas as pd
from tqdm import tqdm
from src.config import REP, TAB, FIG
from src.logging_utils import get_logger, Timer
from src import dataset, baselines, metrics

def build_eval(panel, log, tag):
    rows = []
    for sid, g in tqdm(panel.groupby("station_id"), desc=f"baseline {tag}"):
        rows.extend(baselines.per_well_eval(dataset.reindex_well(g)))
    df = pd.DataFrame(rows, columns=["station_id", "split", "h", "model", "y", "yhat", "scale"])
    log.info(f"[{tag}] eval rows={len(df)}")
    summ = metrics.summarize(df)
    summ.to_csv(TAB / f"baseline_metrics_{tag}.csv", index=False)
    log.info(f"[{tag}] metrics:\n{summ.to_string(index=False)}")
    return summ

def main():
    log = get_logger("cycle4")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    panel = dataset.load_panel()
    nat = dataset.cohort(panel, min_q=16)
    igp = dataset.cohort(panel, min_q=16, region="igp")
    log.info(f"national wells={nat.station_id.nunique()} | IGP wells={igp.station_id.nunique()}")

    with Timer(log, "baselines national"):
        s_nat = build_eval(nat, log, "national")
    with Timer(log, "baselines IGP"):
        s_igp = build_eval(igp, log, "igp")

    # figure: national test RMSE vs horizon
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = s_nat[s_nat.split == "test"]
    plt.figure()
    for m, gg in t.groupby("model"):
        plt.plot(gg["h"], gg["RMSE"], marker="o", label=m)
    plt.xlabel("horizon (quarters)"); plt.ylabel("RMSE (m)")
    plt.title("National test: baseline RMSE vs horizon"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "baseline_rmse_vs_horizon.png", dpi=150); plt.close()

    out = {"national": s_nat.to_dict("records"), "igp": s_igp.to_dict("records")}
    with open(REP / "cycle4_summary.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"WROTE cycle4_summary.json:\n{json.dumps(out, indent=2, default=str)}")
    log.info("CYCLE 4 COMPLETE")

if __name__ == "__main__":
    main()
