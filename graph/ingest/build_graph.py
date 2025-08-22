import os
import csv
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple

import pandas as pd
from tqdm import tqdm


# Paths within the monorepo
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_CSV = os.path.join(REPO_ROOT, "data", "bible_berean_translation - bible.csv")
FULFILLMENT_MD = os.path.join(REPO_ROOT, "data", "bible_fulfillment.md")
OUT_DIR = os.path.join(REPO_ROOT, "results", "graph", "csv")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def to_osis(book: str, chapter: int, verse: int) -> str:
    # Simple OSIS-like id (book abbreviations could be refined later)
    return f"{book}.{chapter}.{verse}"


def normalize_book(book: str) -> str:
    return book.replace(" ", "_")


def export_nodes() -> Tuple[str, str, str]:
    df = pd.read_csv(DATA_CSV)
    # Expect columns: reference, text, chapter, book, testament
    books = (
        df[["book", "testament"]]
        .drop_duplicates()
        .sort_values(["testament", "book"]) 
    )
    chapters = (
        df[["book", "chapter"]]
        .drop_duplicates()
        .sort_values(["book", "chapter"]) 
    )

    book_csv = os.path.join(OUT_DIR, "books.csv")
    with open(book_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "testament"])
        for _, row in books.iterrows():
            book_id = row["book"]
            w.writerow([book_id, row["book"], row["testament"]])

    chapter_csv = os.path.join(OUT_DIR, "chapters.csv")
    with open(chapter_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "chapterNumber", "bookId"])
        for _, row in chapters.iterrows():
            chap_id = f"{row['book']}.{int(row['chapter'])}"
            w.writerow([chap_id, int(row["chapter"]), row["book"]])

    verse_csv = os.path.join(OUT_DIR, "verses.csv")
    with open(verse_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "reference", "text", "chapterId", "bookId"])
        for idx, row in df.iterrows():
            # reference looks like "Genesis 1:1" → parse verse number
            try:
                ref = row["reference"]
                verse_num = int(str(ref).split(":")[1])
            except Exception:
                verse_num = 1
            chap_id = f"{row['book']}.{int(row['chapter'])}"
            osis = to_osis(row["book"], int(row["chapter"]), int(verse_num))
            w.writerow([osis, row["reference"], row["text"], chap_id, row["book"]])

    return book_csv, chapter_csv, verse_csv


def export_hierarchy_edges() -> Tuple[str, str]:
    df = pd.read_csv(DATA_CSV)
    chapters = (
        df[["book", "chapter"]]
        .drop_duplicates()
        .sort_values(["book", "chapter"]) 
    )
    contains_bc = os.path.join(OUT_DIR, "book_contains_chapter.csv")
    with open(contains_bc, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE"])
        for _, row in chapters.iterrows():
            w.writerow([row["book"], f"{row['book']}.{int(row['chapter'])}", "CONTAINS"])

    contains_cv = os.path.join(OUT_DIR, "chapter_contains_verse.csv")
    with open(contains_cv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE"])
        for _, row in df.iterrows():
            try:
                ref = row["reference"]
                verse_num = int(str(ref).split(":")[1])
            except Exception:
                verse_num = 1
            chap_id = f"{row['book']}.{int(row['chapter'])}"
            osis = to_osis(row["book"], int(row["chapter"]), int(verse_num))
            w.writerow([chap_id, osis, "CONTAINS"])

    return contains_bc, contains_cv


def main() -> None:
    ensure_dirs()
    print("Exporting nodes…")
    export_nodes()
    print("Exporting hierarchy edges…")
    export_hierarchy_edges()
    print(f"Done. CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()


