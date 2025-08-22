Ingestion overview

1) Export node and relationship CSVs using the Python scripts in this folder.
   - Verse OSIS id: Book.Chapter.Verse (e.g., Gen.1.1)
   - Files written under ../../results/graph/csv/

2) Start Neo4j:

   docker compose -f ../docker-compose.yml up -d

3) Load schema:

   Open Neo4j Browser at http://localhost:7474 and run the statements in ../schema/constraints.cypher

4) Bulk import (fastest for fresh DB) or Cypher LOAD CSV:
   - Use the provided loader scripts (`load_csv.cypher`) or the Python Neo4j driver to create nodes/edges.


