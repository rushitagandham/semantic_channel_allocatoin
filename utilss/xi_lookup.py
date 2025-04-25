# utils/xi_lookup.py
import json
import numpy as np

_LOOKUP = None           # loaded lazily

def _load_table():
    global _LOOKUP
    with open("data/xi_table.json") as f:
        raw = json.load(f)
    _LOOKUP = {
        tuple(map(int, key.split(","))): float(val)
        for key, val in raw.items()
    }

def xi_lookup(k: int, snr_db: float) -> float:
    """
    Return expected semantic-similarity ξ for (k symbols/word, SNR_dB).
    Uses nearest-neighbour if exact key not stored.
    """
    if _LOOKUP is None:
        _load_table()

    snr_rounded = int(round(snr_db))          #  e.g. 7.3 → 7 dB
    key = (int(k), snr_rounded)
    if key in _LOOKUP:
        return _LOOKUP[key]

    # nearest neighbour fallback
    ks   = np.array([p[0]   for p in _LOOKUP.keys()])
    snrs = np.array([p[1]   for p in _LOOKUP.keys()])
    idx  = np.argmin((ks-k)**2 + (snrs-snr_rounded)**2)
    return list(_LOOKUP.values())[idx]
