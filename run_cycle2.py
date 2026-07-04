import json, platform
from src.config import PROC, REP
from src.logging_utils import get_logger, Timer
from src import diagnostics
import pandas as pd

def main():
    log = get_logger("cycle2")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    with Timer(log, "load parquet"):
        df = pd.read_parquet(PROC / "raw_combined.parquet")
        log.info(f"loaded {df.shape}")
    with Timer(log, "DIAGNOSTICS"):
        summary = diagnostics.run(df, log)
    with open(REP / "cycle2_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"WROTE cycle2_summary.json:\n{json.dumps(summary, indent=2, default=str)}")
    log.info("CYCLE 2 COMPLETE")

if __name__ == "__main__":
    main()
