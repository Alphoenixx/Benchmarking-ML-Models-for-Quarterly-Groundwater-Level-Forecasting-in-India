import kagglehub
import shutil
import glob
from pathlib import Path

def download_dataset():
    print("Attempting to download via kagglehub...")
    try:
        path = kagglehub.dataset_download("kushvinthmadhavan/india-groundwater-climate-time-series-19942025")
        print("Path to dataset files:", path)
        raw_dir = Path("data/raw")
        
        for f in glob.glob(path + "/*.csv"):
            shutil.copy(f, raw_dir / Path(f).name)
            print("Copied", f, "to", raw_dir)
    except Exception as e:
        print("Error downloading via kagglehub:", e)

if __name__ == "__main__":
    download_dataset()
