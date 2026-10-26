"""Step 4: Admaster XLSX Label Reader, Normalization & Model B Prior Filler.

Owner: Freshman 4
Responsibilities:
  1. Read `admaster.xlsx` (works with or without openpyxl installed).
  2. Fill in baseline Arousal, Dominance, and Keywords for the 3 initial ads
     if blank, and normalize Pleasure/Arousal/Dominance from [-100, 100] to [-1.0, 1.0].
  3. Join normalized PAD targets onto the epoch feature DataFrame.
"""

from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ADMASTER = BASE_DIR / "admaster.xlsx"

# Fallback consensus priors for the 3 sample ads when Arousal/Dominance are blank in Excel
DEFAULT_PRIORS = {
    "muse_20261019_165020.csv": {
        "Arousal": 60.0,
        "dominance": 45.0,
        "Keywords": "upbeat, engaging commercial, positive brand appeal",
    },
    "muse_20261019_165159.csv": {
        "Arousal": 40.0,
        "dominance": 30.0,
        "Keywords": "short-form reel, moderate energy, casual product showcase",
    },
    "muse_20261019_165536.csv": {
        "Arousal": 55.0,
        "dominance": -35.0,
        "Keywords": "tense, negative valence, dissonant / aversive tone",
    },
}


def _read_xlsx_stdlib(xlsx_path: Path) -> pd.DataFrame:
    """Fallback XLSX parser using Python standard library zipfile + xml."""
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(xlsx_path) as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            tree = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in tree.findall(f"{ns}si"):
                text = "".join(n.text for n in si.iter(f"{ns}t") if n.text)
                shared_strings.append(text)

        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        rows_data = []
        for row in sheet.iter(f"{ns}row"):
            row_dict = {}
            for c in row.iter(f"{ns}c"):
                ref = c.attrib.get("r", "")
                col_letter = "".join(ch for ch in ref if ch.isalpha())
                t_attr = c.attrib.get("t")
                v = c.find(f"{ns}v")
                val = v.text if v is not None else None
                if t_attr == "s" and val is not None:
                    val = shared_strings[int(val)]
                row_dict[col_letter] = val
            if row_dict:
                rows_data.append(row_dict)

    headers = rows_data[0]
    records = []
    for r in rows_data[1:]:
        rec = {headers[col]: r.get(col) for col in headers}
        records.append(rec)
    return pd.DataFrame(records)


def load_admaster_labels(xlsx_path: Path = DEFAULT_ADMASTER) -> pd.DataFrame:
    """Load `admaster.xlsx`, fill missing PAD priors, and scale to [-1.0, 1.0]."""
    try:
        df = pd.read_excel(xlsx_path)
    except Exception:
        df = _read_xlsx_stdlib(xlsx_path)

    for idx, row in df.iterrows():
        eeg_id = str(row["EEGid"]).strip()
        prior = DEFAULT_PRIORS.get(eeg_id, {})

        if pd.isna(row.get("Arousal")) or str(row.get("Arousal", "")).strip() == "":
            df.at[idx, "Arousal"] = prior.get("Arousal", 0.0)
        if pd.isna(row.get("dominance")) or str(row.get("dominance", "")).strip() == "":
            df.at[idx, "dominance"] = prior.get("dominance", 0.0)
        if pd.isna(row.get("Keywords")) or str(row.get("Keywords", "")).strip() == "":
            df.at[idx, "Keywords"] = prior.get("Keywords", "general stimulus")

    df["Pleasure"] = pd.to_numeric(df["Pleasure"], errors="coerce").fillna(0.0)
    df["Arousal"] = pd.to_numeric(df["Arousal"], errors="coerce").fillna(0.0)
    df["dominance"] = pd.to_numeric(df["dominance"], errors="coerce").fillna(0.0)

    # Normalize [-100, 100] scale to [-1.0, 1.0]
    df["P_norm"] = (df["Pleasure"] / 100.0).clip(-1.0, 1.0)
    df["A_norm"] = (df["Arousal"] / 100.0).clip(-1.0, 1.0)
    df["D_norm"] = (df["dominance"] / 100.0).clip(-1.0, 1.0)

    return df
