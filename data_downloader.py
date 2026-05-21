import requests
import zipfile
from pathlib import Path

BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"
DATASET_NAME = "nfcorpus"

class Downloader:
    def download_file(self, url: str, output_path: str) -> None:
        if Path(output_path).exists():
            return
        
        with requests.get(url, stream=True) as r:
            r.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
    def extract_zip(self, zip_path: str, extract_to: str = "."):        
        stem = Path(zip_path).stem
        
        if Path(stem).exists():
            Path(stem).rmdir()
        
        with zipfile.ZipFile(zip_path, "r") as zip:
            zip.extractall(extract_to)

if __name__ == "__main__":
    print("--- Downloading NFCorpus (BEIR) ---")

    zip_url = BASE_URL + f"{DATASET_NAME}.zip"
    zip_path = f"{DATASET_NAME}.zip"

    try:
        downloader = Downloader()
        downloader.download_file(zip_url, zip_path)
        downloader.extract_zip(zip_path)
        print(f"[Success] Saved.")
    except Exception as e:
        print(f"[Error] {e}")
    finally:
        if Path(zip_path).exists():
            Path(zip_path).unlink()
            print(f"[Cleanup] Zipfile removed.")