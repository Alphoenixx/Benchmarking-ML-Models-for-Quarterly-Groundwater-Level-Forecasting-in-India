import pandas as pd, glob
from src.config import RAW, PROC
from src.logging_utils import get_logger, Timer

def load_raw():
    log = get_logger("cycle1")
    files = glob.glob(str(RAW / "*.csv"))
    log.info(f"Found {len(files)} CSV(s) in data/raw: {files}")
    if not files:
        raise FileNotFoundError("Put the Kaggle India groundwater CSV in data/raw/")
    frames = []
    for f in files:
        with Timer(log, f"read {f}"):
            df = pd.read_csv(f, low_memory=False)
            log.info(f"  shape={df.shape}  cols={list(df.columns)[:20]}")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    log.info(f"Combined shape={df.shape}")
    # >>> IMPORTANT: print dtypes + head so we can map columns next cycle
    log.info(f"dtypes:\n{df.dtypes}")
    log.info(f"head:\n{df.head(5).to_string()}")
    df.to_parquet(PROC / "raw_combined.parquet", index=False)
    return df
