import os
import csv
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_CSV = os.path.join(REPO_ROOT, "data", "bible_berean_translation - bible.csv")
OUT_DIR = os.path.join(REPO_ROOT, "results", "graph", "csv")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def to_osis(book: str, chapter: int, verse: int) -> str:
    return f"{book}.{chapter}.{verse}"


def embed_verses(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available. Install extras in pyproject.")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def top_k_similar(emb: np.ndarray, k: int = 3, min_cos: float = 0.83) -> List[List[Tuple[int, float]]]:
    # Cosine similarities of each row to all rows; process in blocks to manage memory
    n = emb.shape[0]
    results: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    block = 2048
    for start in tqdm(range(0, n, block), desc="similarity"):
        end = min(n, start + block)
        sims = emb[start:end] @ emb.T
        for i in range(end - start):
            row = sims[i]
            row[start + i] = -1.0  # exclude self
            idxs = np.argpartition(-row, k)[:k]
            scored = [(int(j), float(row[j])) for j in idxs if row[j] >= min_cos]
            scored.sort(key=lambda t: t[1], reverse=True)
            results[start + i] = scored[:k]
    return results


def build_reference_edges(k: int = 3, min_cos: float = 0.83, model_name: str = "all-MiniLM-L6-v2") -> str:
    df = pd.read_csv(DATA_CSV)
    ids = []
    texts = []
    for _, row in df.iterrows():
        ref = row["reference"]
        try:
            verse_num = int(str(ref).split(":")[1])
        except Exception:
            verse_num = 1
        ids.append(to_osis(row["book"], int(row["chapter"]), int(verse_num)))
        texts.append(str(row["text"]))

    emb = embed_verses(texts, model_name=model_name)
    neighbors = top_k_similar(emb, k=k, min_cos=min_cos)

    out_csv = os.path.join(OUT_DIR, "references_edges.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE", "score", "method"])  # directed edges
        for i, nbrs in enumerate(neighbors):
            for j, score in nbrs:
                w.writerow([ids[i], ids[j], "REFERENCES", round(float(score), 6), "embedding_sbert"])    
    return out_csv


if __name__ == "__main__":
    ensure_dirs()
    print("Computing embeddings and neighbor edges…")
    path = build_reference_edges()
    print(f"Wrote {path}")


