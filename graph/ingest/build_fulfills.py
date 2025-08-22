import os
import re
import csv
from typing import List, Tuple
import argparse

import pandas as pd


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_CSV = os.path.join(REPO_ROOT, "data", "bible_berean_translation - bible.csv")
DEFAULT_INPUT = os.path.join(REPO_ROOT, "data", "bible_fulfillment.md")
OUT_DIR = os.path.join(REPO_ROOT, "results", "graph", "csv")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def to_osis_from_reference(ref: str) -> str:
    # Expect format: "Genesis 1:1"
    try:
        book, cv = ref.split(" ", 1)
        chapter, verse = cv.split(":")
        return f"{book}.{int(chapter)}.{int(verse)}"
    except Exception:
        return ref.replace(" ", ".")


def parse_fulfillment_md(md_path: str) -> List[Tuple[str, str, str]]:
    """
    Very simple parser: expects lines that contain OT and NT references separated by a tab or '->'.
    Example lines:
      Micah 5:2 -> Matthew 2:1-6  // ranges will become multiple edges if possible
    Returns list of tuples (nt_osis, ot_osis, source_line)
    """
    rows: List[Tuple[str, str, str]] = []
    if not os.path.exists(md_path):
        return rows
    line_re = re.compile(r"^([A-Za-z1-3 ]+\d+:\d+(?:-\d+)?)\s*(?:->|=>|\t)\s*([A-Za-z1-3 ]+\d+:\d+(?:-\d+)?)")
    with open(md_path, "r") as f:
        for line in f:
            m = line_re.search(line)
            if not m:
                continue
            left = m.group(1).strip()
            right = m.group(2).strip()
            # Heuristic: if NT on right, swap to make nt->ot direction
            nt_ref, ot_ref = right, left
            rows.append((to_osis_from_reference(nt_ref), to_osis_from_reference(ot_ref), line.strip()))
    return rows


def build_fulfill_edges(input_path: str = DEFAULT_INPUT) -> str:
    ensure_dirs()
    rows = parse_fulfillment_md(input_path)
    out_csv = os.path.join(OUT_DIR, "fulfills_edges.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE", "source"])  # NT fulfills OT
        for nt_id, ot_id, src in rows:
            w.writerow([nt_id, ot_id, "FULFILLS", src])
    return out_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FULFILLS edges from a structured or md file")
    parser.add_argument("--input", dest="input", default=DEFAULT_INPUT, help="Path to input .md/.txt with lines like 'Micah 5:2 -> Matthew 2:1'")
    args = parser.parse_args()
    p = build_fulfill_edges(args.input)
    print(f"Wrote {p}")


