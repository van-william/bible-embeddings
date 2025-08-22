import os
import csv
from typing import List, Tuple

import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_CSV = os.path.join(REPO_ROOT, "data", "bible_berean_translation - bible.csv")
OUT_DIR = os.path.join(REPO_ROOT, "results", "graph", "csv")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def to_osis(book: str, chapter: int, verse: int) -> str:
    return f"{book}.{chapter}.{verse}"


def normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2019", "'")
        .replace("—", "-")
    )


def build_quote_edges(min_ratio: int = 92) -> str:
    df = pd.read_csv(DATA_CSV)

    # Build a mapping from verse id to normalized text
    verse_rows = []
    for _, row in df.iterrows():
        ref = row["reference"]
        try:
            verse_num = int(str(ref).split(":")[1])
        except Exception:
            verse_num = 1
        verse_id = to_osis(row["book"], int(row["chapter"]), int(verse_num))
        verse_rows.append((verse_id, normalize_text(str(row["text"]))))

    verse_ids = [vid for vid, _ in verse_rows]
    texts = [t for _, t in verse_rows]

    # For each NT verse, attempt to find OT verses that are near-exact quotes
    df_nt = df[df["testament"] == "NT"].copy()

    out_csv = os.path.join(OUT_DIR, "quotes_edges.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE", "ratio", "method"])  # START quotes END (OT source)

        for _, row in tqdm(df_nt.iterrows(), total=len(df_nt)):
            ref = row["reference"]
            try:
                verse_num = int(str(ref).split(":")[1])
            except Exception:
                verse_num = 1
            nt_id = to_osis(row["book"], int(row["chapter"]), int(verse_num))
            nt_text = normalize_text(str(row["text"]))

            # Find best matches over all verses; restrict to OT hits in post-processing
            matches = process.extract(nt_text, texts, scorer=fuzz.token_set_ratio, limit=5)
            for candidate_text, ratio, idx in matches:
                if ratio < min_ratio:
                    continue
                ot_id = verse_ids[idx]
                # Skip self-matches
                if ot_id == nt_id:
                    continue
                w.writerow([nt_id, ot_id, "QUOTES", int(ratio), "string_match"])

    return out_csv


if __name__ == "__main__":
    ensure_dirs()
    path = build_quote_edges()
    print(f"Wrote {path}")


