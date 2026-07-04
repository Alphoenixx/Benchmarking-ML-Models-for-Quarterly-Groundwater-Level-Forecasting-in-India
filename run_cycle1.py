import json, platform
from src.config import REP, TARGET_STATES
from src.logging_utils import get_logger, Timer
from src import data_ingest, eda

def main():
    log = get_logger("cycle1")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    summary = {"target_states": TARGET_STATES}
    with Timer(log, "INGEST"):
        df = data_ingest.load_raw()
        summary["raw_shape"] = list(df.shape)
    with Timer(log, "EDA"):
        summary.update(eda.run(df, log))   # eda.run returns a dict of key stats
    with open(REP / "cycle1_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"WROTE cycle1_summary.json:\n{json.dumps(summary, indent=2, default=str)}")
    log.info("CYCLE 1 COMPLETE")

if __name__ == "__main__":
    main()
