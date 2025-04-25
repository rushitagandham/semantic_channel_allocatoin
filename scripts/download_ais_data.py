import os
import requests
from tqdm import tqdm
import zipfile
import time

def download_ais_sample(save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2022/AIS_2022_01_01.zip"
    local_zip_path = os.path.join(save_dir, "AIS_2022_01_01.zip")

    print(f"Downloading AIS data from {url}...")
    
    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))

            with open(local_zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    except Exception as e:
        print(f"[ERROR] Chunked download failed: {e}")
        print("Retrying download with fallback (non-stream)...")
        time.sleep(3)

        try:
            response = requests.get(url, timeout=60)
            with open(local_zip_path, 'wb') as f:
                f.write(response.content)
            print("Downloaded successfully using fallback method.")

        except Exception as e:
            print("[CRITICAL] Download failed again.")
            print(e)
            return

    # Unzipping
    print("Unzipping...")
    try:
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        print("✅ Unzip complete!")
    except zipfile.BadZipFile:
        print("❌ Error: Corrupted ZIP file. Please delete and try again.")

if __name__ == "__main__":
    download_ais_sample()
