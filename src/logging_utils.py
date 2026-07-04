import logging, sys, time
from datetime import datetime
from src.config import LOG

def get_logger(name="cycle"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    sh = logging.StreamHandler(sys.stdout)   # console
    sh.setFormatter(fmt); sh.flush = sys.stdout.flush
    fh = logging.FileHandler(LOG / f"{name}_{ts}.log", encoding="utf-8")  # file
    fh.setFormatter(fmt)

    logger.addHandler(sh); logger.addHandler(fh)
    logger.info(f"Logger started -> {LOG / f'{name}_{ts}.log'}")
    return logger

class Timer:
    def __init__(self, logger, label): self.logger, self.label = logger, label
    def __enter__(self): self.t = time.time(); self.logger.info(f"START: {self.label}"); return self
    def __exit__(self, *a): self.logger.info(f"END:   {self.label} ({time.time()-self.t:.2f}s)")
