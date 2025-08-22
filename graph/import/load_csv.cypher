// Assumes CSVs are placed in /var/lib/neo4j/import via docker volume mapping

// Nodes
LOAD CSV WITH HEADERS FROM 'file:///csv/books.csv' AS row
MERGE (b:Book {id: row.id})
  SET b.name = row.name, b.testament = row.testament;

LOAD CSV WITH HEADERS FROM 'file:///csv/chapters.csv' AS row
MERGE (c:Chapter {id: row.id})
  SET c.chapterNumber = toInteger(row.chapterNumber)
MERGE (b:Book {id: row.bookId})
MERGE (b)-[:CONTAINS]->(c);

LOAD CSV WITH HEADERS FROM 'file:///csv/verses.csv' AS row
MERGE (v:Verse {id: row.id})
  SET v.reference = row.reference, v.text = row.text
MERGE (c:Chapter {id: row.chapterId})
MERGE (b:Book {id: row.bookId})
MERGE (c)-[:CONTAINS]->(v)
MERGE (b)-[:CONTAINS]->(c);

// Edges
LOAD CSV WITH HEADERS FROM 'file:///csv/quotes_edges.csv' AS row
MATCH (a:Verse {id: row.`:START_ID`}), (b:Verse {id: row.`:END_ID`})
MERGE (a)-[r:QUOTES]->(b)
  SET r.ratio = toInteger(row.ratio), r.method = row.method;

LOAD CSV WITH HEADERS FROM 'file:///csv/references_edges.csv' AS row
MATCH (a:Verse {id: row.`:START_ID`}), (b:Verse {id: row.`:END_ID`})
MERGE (a)-[r:REFERENCES]->(b)
  SET r.score = toFloat(row.score), r.method = row.method;

LOAD CSV WITH HEADERS FROM 'file:///csv/fulfills_edges.csv' AS row
MATCH (a:Verse {id: row.`:START_ID`}), (b:Verse {id: row.`:END_ID`})
MERGE (a)-[r:FULFILLS]->(b)
  SET r.source = row.source;


