import json, platform
import pandas as pd
from src.config import PROC, REP
from src.logging_utils import get_logger, Timer
from src import preprocess

def main():
    log = get_logger("cycle3")
    log.info(f"Platform: {platform.platform()} | Python {platform.python_version()}")
    df = pd.read_parquet(PROC / "raw_combined.parquet")
    log.info(f"loaded raw {df.shape}")
    with Timer(log, "PREPROCESS"):
        summary = preprocess.run(df, log)
    with open(REP / "cycle3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"WROTE cycle3_summary.json:\n{json.dumps(summary, indent=2, default=str)}")
    log.info("CYCLE 3 COMPLETE")

if __name__ == "__main__":
    main()
