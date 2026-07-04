from pathlib import Path

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
LOG = ROOT / "outputs" / "logs"
REP = ROOT / "outputs" / "reports"
for d in (RAW, PROC, FIG, TAB, LOG, REP):
    d.mkdir(parents=True, exist_ok=True)

# Target region filter (adjust column names after first inspection)
TARGET_STATES = ["Punjab", "Haryana", "Uttar Pradesh", "Delhi", "Chandigarh"]
