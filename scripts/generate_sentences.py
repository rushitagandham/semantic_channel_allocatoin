import os
import pandas as pd
from tqdm import tqdm
import re

RAW_DIR = os.path.join("data", "raw")
SAVE_DIR = os.path.join("data", "processed")
os.makedirs(SAVE_DIR, exist_ok=True)

def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s\.,:-]', '', text).strip()
    return ""

def generate_sentence(row):
    name = clean_text(row.get('VesselName', 'Unknown'))
    dest = clean_text(row.get('Destination', 'Unknown'))
    sog = row.get('SOG', 0.0)  # Speed over ground
    eta = clean_text(row.get('ETA', 'Unknown'))
    lat = round(row.get('LAT', 0.0), 2)
    lon = round(row.get('LON', 0.0), 2)
    vessel_type = clean_text(row.get('VesselType', 'vessel'))

    # You can customize or add more templates
    if sog == 0.0:
        return f"{vessel_type} {name} is stationary near ({lat} N, {lon} W)."
    elif pd.notna(dest):
        return f"{vessel_type} {name} heading to {dest} at {sog:.1f} knots, ETA {eta}."
    else:
        return f"{vessel_type} {name} cruising at {sog:.1f} knots near ({lat} N, {lon} W)."

def generate_sentences_from_csv(csv_path):
    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Rows found: {len(df)}")

    # Drop rows with missing crucial fields
    df = df.dropna(subset=["VesselName", "LAT", "LON"])
    sentences = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            sentence = generate_sentence(row)
            sentences.append(sentence)
        except:
            continue

    # Save .txt file
    txt_path = os.path.join(SAVE_DIR, "maritime_sentences.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    print(f"✅ Generated {len(sentences)} sentences.")
    print(f"Saved to {txt_path}")

if __name__ == "__main__":
    # Replace with your actual file name if different
    file_name = "AIS_2022_01_01.csv"
    csv_path = os.path.join(RAW_DIR, file_name)
    generate_sentences_from_csv(csv_path)
