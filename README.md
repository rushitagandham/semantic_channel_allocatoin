## 🔁 One-Click Setup (Full Bash Workflow)

Get the full **Semantic Maritime Communication System** up and running in a few simple commands!  
Everything from environment setup to evaluation — just copy, paste, and go 💻✨

---

### 🛠️ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/semantic_communication_system.git
cd semantic_communication_system
```

---

### 🐍 Step 2: Set Up the Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate         # For Windows PowerShell
```

---

### 📦 Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 🌍 Step 4: Download AIS Dataset

```bash
python scripts/download_ais_data.py
```

This will download and unzip **AIS vessel data** from NOAA’s MarineCadastre.gov.

---

### ✍️ Step 5: Generate Maritime Sentences

```bash
python scripts/generate_sentences.py
```

Converts AIS CSV data into structured natural language sentences like:

> `Cargo vessel Blue Whale heading to Port of LA at 12.5 knots.`

---

### 🔤 Step 6: Build the Tokenizer Vocabulary

```bash
python utils/build_vocab.py
```

Builds the vocabulary from generated maritime messages.

---
### Step 7: Build the Semantic Spectral Efficiency (SSE) Lookup Table

```bash
python scripts/build_xi_table.py
```
Generate a lookup table (xi_table.json) that maps different values of symbol length ($k$) and SNR to their corresponding Semantic Spectral Efficiency (SSE). This table is used during training and evaluation for optimal channel allocation.

### 🧠 Step 8: Train the Semantic Communication Model

```bash
python scripts/train.py
```

This trains a transformer + CNN pipeline that encodes, transmits (through a noisy channel), and decodes maritime messages.

---

### 🧪 Step 9: Evaluate Model Performance

```bash
python scripts/evaluate.py
```

You’ll see outputs like:

```text
--- SAMPLE ---
Ref : Cargo ship heading to Port LA at 15 knots.
Pred: Cargo vessel en route to Port LA at 15 knots.
BLEU: 0.71 | Semantic Sim: 0.93
```

---

> ⚠️ **Note**:  
> This repo excludes large raw data and model files for size reasons.  
> You can regenerate everything using the commands above.

---

  
📜 Licensed under MIT 
