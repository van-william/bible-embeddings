import os
import csv
from typing import List, Tuple
from urllib.parse import urlparse
import ssl

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import pg8000

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
OUT_DIR = os.path.join(REPO_ROOT, "results", "graph", "csv")


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def parse_pgvector_text(text: str) -> List[float]:
    # pgvector ::text returns e.g. '[0.1, 0.2, ...]'
    text = text.strip().strip("[]")
    if not text:
        return []
    return [float(x) for x in text.split(",")]


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def top_k_similar(emb: np.ndarray, k: int, min_cos: float) -> List[List[Tuple[int, float]]]:
    n = emb.shape[0]
    result: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    sims = emb @ emb.T
    for i in range(n):
        sims[i, i] = -1.0
        idxs = np.argpartition(-sims[i], k)[:k]
        pairs = [(int(j), float(sims[i, j])) for j in idxs if sims[i, j] >= min_cos]
        pairs.sort(key=lambda t: t[1], reverse=True)
        result[i] = pairs[:k]
    return result


def _connect_pg8000_from_url(url: str):
    u = urlparse(url)
    # Handle postgres or postgresql scheme
    if u.scheme not in ("postgres", "postgresql"):
        raise RuntimeError(f"Unsupported DB URL scheme: {u.scheme}")
    ssl_ctx = ssl.create_default_context()
    return pg8000.connect(
        host=u.hostname,
        port=u.port or 5432,
        database=u.path.lstrip("/") or "postgres",
        user=u.username,
        password=u.password,
        ssl_context=ssl_ctx,
    )


def fetch_embeddings(conn_str: str, table: str, id_sql: str) -> Tuple[List[str], np.ndarray]:
    conn = _connect_pg8000_from_url(conn_str)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {id_sql}, embedding::text FROM {table} ORDER BY 1;")
            rows = cur.fetchall()
    finally:
        conn.close()
    ids = [r[0] for r in rows]
    vecs = [parse_pgvector_text(r[1]) for r in rows]
    emb = np.array(vecs, dtype=np.float32)
    emb = normalize_rows(emb)
    return ids, emb


def write_edges(out_path: str, ids: List[str], neighbors: List[List[Tuple[int, float]]], method: str) -> None:
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE", "score", "method"])  # directed
        for i, nbrs in enumerate(neighbors):
            for j, score in nbrs:
                w.writerow([ids[i], ids[j], "REFERENCES", round(float(score), 6), method])


def main() -> None:
    ensure_dirs()
    # Load .env from repo root if present
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        raise RuntimeError("DATABASE_URL not set; put it in .env or export it in your shell")

    # Chapters → use ids like 'Book.Chapter' to match graph Chapter ids
    chap_ids, chap_emb = fetch_embeddings(
        conn_str,
        table="chapter_chunks",
        id_sql="(book || '.' || chapter) as id"
    )
    chap_neighbors = top_k_similar(chap_emb, k=3, min_cos=0.83)
    write_edges(os.path.join(OUT_DIR, "chapter_references_edges.csv"), chap_ids, chap_neighbors, "pgvector_chapter")

    # Books → use book name ids to match graph Book ids
    book_ids, book_emb = fetch_embeddings(
        conn_str,
        table="book_chunks",
        id_sql="book as id"
    )
    book_neighbors = top_k_similar(book_emb, k=3, min_cos=0.83)
    write_edges(os.path.join(OUT_DIR, "book_references_edges.csv"), book_ids, book_neighbors, "pgvector_book")

    print(f"Wrote chapter and book reference edges to {OUT_DIR}")


if __name__ == "__main__":
    main()


