Bible Graph (Neo4j)

What this adds
- Hierarchy: `Book` → `Chapter` → `Verse`
- Relationships: `QUOTES`, `REFERENCES`, `FULFILLS`

Quick start

1) Start Neo4j

```
cd graph
docker compose up -d
```

2) Generate CSVs

```
python ingest/build_graph.py
python ingest/build_quotes.py
# python ingest/build_references.py  # optional; heavier (embeddings)
python ingest/build_fulfills.py
```

CSV output: `results/graph/csv/` → copy into `graph/import/csv/` to load.

3) Load schema and data in Browser (http://localhost:7474)
- Run the statements from `schema/constraints.cypher`
- Then run the statements in `ingest/load_csv.cypher`

Login to Neo4j Browser
- URL: open `http://localhost:7474`
- Connect URL: `neo4j://localhost:7687`
- Auth: Username `neo4j`, Password `password`
- Change password (optional):
  ```
  docker exec bible_neo4j cypher-shell -u neo4j -p password "ALTER CURRENT USER SET PASSWORD FROM 'password' TO 'newpass';"
  ```

Use existing Neon embeddings (optional)
If you already have embeddings in your Neon database (from `openai_embed_books.py` or `bible_embeddings.py`):

1) Ensure `DATABASE_URL` is exported in your shell.
2) Build reference edges from Neon:
   ```
   uv run python graph/ingest/build_references_from_neon.py
   ```
   This writes:
   - `results/graph/csv/chapter_references_edges.csv` → `(:Chapter)-[:REFERENCES]->(:Chapter)`
   - `results/graph/csv/book_references_edges.csv` → `(:Book)-[:REFERENCES]->(:Book)`
3) Load into Neo4j:
   ```
   cp results/graph/csv/*references*.csv graph/import/csv/
   docker cp graph/import/csv/. bible_neo4j:/var/lib/neo4j/import/csv
   docker exec bible_neo4j cypher-shell -u neo4j -p password -f /var/lib/neo4j/import/load_csv.cypher | cat
   ```

Notes
- Verse IDs: `Book.Chapter.Verse` (e.g., `Genesis.1.1`).
- REFERENCES defaults: cosine ≥ 0.83, top-3 per verse (edit in `ingest/build_references.py`).
- QUOTES defaults: token_set_ratio ≥ 92 (edit in `ingest/build_quotes.py`).

Curated FULFILLS edges (Markdown format)

Add mappings in `data/bible_fulfillment.md`. Each line creates one or more NT→OT edges.

Format
- Single verse → verse:
  - `Micah 5:2 -> Matthew 2:1`
- Verse → verse range (creates multiple edges):
  - `Zechariah 9:9 -> Matthew 21:4-5`
- Tab is also supported instead of `->`.

Direction
- Always NT fulfills OT: `NT -> OT` produces edge `(:Verse {NT})-[:FULFILLS]->(:Verse {OT})`.

Build and load
```
# Option A: auto-extract structured lines from long markdown
uv run python ingest/normalize_fulfillment_md.py
uv run python ingest/build_fulfills.py --input ../../data/bible_fulfillment_structured.txt

# Option B: you already maintain simple "A -> B" lines
uv run python ingest/build_fulfills.py
cp ../../results/graph/csv/fulfills_edges.csv import/csv/
docker cp import/csv/fulfills_edges.csv bible_neo4j:/var/lib/neo4j/import/csv/
docker exec bible_neo4j cypher-shell -u neo4j -p password -f /var/lib/neo4j/import/load_csv.cypher | cat
```

Verify
```
docker exec bible_neo4j cypher-shell -u neo4j -p password "MATCH ()-[r:FULFILLS]->() RETURN count(r);" | cat
```

Cypher quick reference ("select *" style)

- All nodes (sample):
  - `MATCH (n) RETURN n LIMIT 100`
- All relationships (sample):
  - `MATCH ()-[r]-() RETURN r LIMIT 100`
- By label:
  - `MATCH (b:Book) RETURN b LIMIT 100`
  - `MATCH (c:Chapter) RETURN c LIMIT 100`
  - `MATCH (v:Verse) RETURN v LIMIT 100`
- With relationship and endpoints:
  - `MATCH (b1:Book)-[r:REFERENCES]->(b2:Book) RETURN b1,r,b2 LIMIT 50`
  - `MATCH (c1:Chapter)-[r:REFERENCES]->(c2:Chapter) RETURN c1,r,c2 LIMIT 50`
  - `MATCH (v1:Verse)-[r:QUOTES]->(v2:Verse) RETURN v1,r,v2 LIMIT 50`
  - `MATCH (v1:Verse)-[r:FULFILLS]->(v2:Verse) RETURN v1,r,v2 LIMIT 50`
- Counts:
  - `MATCH (n:Book) RETURN count(n)`
  - `MATCH (n:Chapter) RETURN count(n)`
  - `MATCH (n:Verse) RETURN count(n)`
  - `MATCH ()-[r:QUOTES]->() RETURN count(r)`
  - `MATCH ()-[r:REFERENCES]->() RETURN count(r)`
  - `MATCH ()-[r:FULFILLS]->() RETURN count(r)`


