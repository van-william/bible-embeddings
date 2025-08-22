// Create uniqueness constraints and useful indexes

// Books
CREATE CONSTRAINT book_id IF NOT EXISTS
FOR (b:Book) REQUIRE b.id IS UNIQUE;
CREATE INDEX book_name IF NOT EXISTS FOR (b:Book) ON (b.name);

// Chapters
CREATE CONSTRAINT chapter_id IF NOT EXISTS
FOR (c:Chapter) REQUIRE c.id IS UNIQUE;
CREATE INDEX chapter_num IF NOT EXISTS FOR (c:Chapter) ON (c.chapterNumber);

// Verses
CREATE CONSTRAINT verse_id IF NOT EXISTS
FOR (v:Verse) REQUIRE v.id IS UNIQUE;
CREATE INDEX verse_ref IF NOT EXISTS FOR (v:Verse) ON (v.reference);

// Relationship type property indexes
CREATE INDEX rel_type IF NOT EXISTS FOR ()-[r:QUOTES]-() ON (r.method);
CREATE INDEX rel_type2 IF NOT EXISTS FOR ()-[r:REFERENCES]-() ON (r.method);
CREATE INDEX rel_type3 IF NOT EXISTS FOR ()-[r:FULFILLS]-() ON (r.source);


