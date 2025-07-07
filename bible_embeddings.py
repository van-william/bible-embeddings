#!/usr/bin/env python3
"""
Script to chunk Bible CSV by chapter and book, then upload embeddings to Neon database.
"""

import os
import pandas as pd
import psycopg2
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
import json

# Load environment variables
load_dotenv()

# Configure Google AI
api_key = os.getenv('VITE_GEMINI_API_KEY')
if not api_key:
    print("❌ VITE_GEMINI_API_KEY not found in environment variables")
    exit(1)

genai.configure(api_key=api_key)

class BibleChunker:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        print(f"📖 Loaded {len(self.df)} verses from {csv_path}")
        
        # Debug: Check testament data
        testament_counts = self.df['testament'].value_counts()
        print(f"📖 Testament distribution: {testament_counts.to_dict()}")
        
        # Check a few specific books
        for book in ['Genesis', 'Psalms', 'Matthew', 'Revelation']:
            if book in self.df['book'].values:
                testament = self.df[self.df['book'] == book]['testament'].iloc[0]
                print(f"📖 {book}: {testament}")
    
    def chunk_by_chapter(self) -> List[Dict[str, Any]]:
        """Group verses by chapter."""
        chunks = []
        
        # Group by book and chapter
        grouped = self.df.groupby(['book', 'chapter'])
        
        for (book, chapter), group in grouped:
            # Combine all verses in the chapter
            verses = group['text'].tolist()
            references = group['reference'].tolist()
            testament = group['testament'].iloc[0]  # All verses in a chapter have same testament
            
            # Create chapter content
            chapter_content = f"Chapter {chapter} of {book}:\n\n"
            for ref, verse in zip(references, verses):
                chapter_content += f"{ref}: {verse}\n\n"
            
            chunks.append({
                'content': chapter_content.strip(),
                'book': str(book),
                'chapter': int(chapter),
                'verse_count': int(len(verses)),
                'verse_references': references,
                'testament': str(testament),
                'chunk_type': 'chapter'
            })
        
        print(f"📚 Created {len(chunks)} chapter chunks")
        return chunks
    
    def chunk_by_book(self) -> List[Dict[str, Any]]:
        """Group verses by book, splitting large books if needed."""
        chunks = []
        
        # Group by book only
        grouped = self.df.groupby('book')
        
        for book, group in grouped:
            # Group by chapter within the book
            chapter_groups = group.groupby('chapter')
            testament = group['testament'].iloc[0]  # All verses in a book have same testament
            
            book_content = f"The Book of {book}:\n\n"
            total_verses = 0
            chapter_count = len(chapter_groups)
            
            # Check if this book is likely to be too large
            estimated_size = len(book_content) + sum(len(chapter_group['text'].sum()) for _, chapter_group in chapter_groups)
            
            if estimated_size > 30000:  # Conservative estimate for 36KB limit
                print(f"📖 Book {book} is large ({estimated_size} chars), splitting by chapters...")
                # Split by chapters for large books
                for chapter, chapter_group in chapter_groups:
                    chapter_content = f"Chapters {chapter} of {book}:\n\n"
                    for _, row in chapter_group.iterrows():
                        chapter_content += f"{row['reference']}: {row['text']}\n"
                    chapter_content += "\n"
                    
                    chunks.append({
                        'content': chapter_content.strip(),
                        'book': str(book),
                        'chapter_count': 1,
                        'verse_count': len(chapter_group),
                        'chunk_type': 'book_chapter',
                        'is_split': True,
                        'original_book': str(book),
                        'testament': str(testament)
                    })
                    total_verses += len(chapter_group)
            else:
                # Regular book chunk
                for chapter, chapter_group in chapter_groups:
                    book_content += f"Chapter {chapter}:\n"
                    for _, row in chapter_group.iterrows():
                        book_content += f"{row['reference']}: {row['text']}\n"
                    book_content += "\n"
                    total_verses += len(chapter_group)
                
                chunks.append({
                    'content': book_content.strip(),
                    'book': str(book),
                    'chapter_count': chapter_count,
                    'verse_count': total_verses,
                    'chunk_type': 'book',
                    'is_split': False,
                    'testament': str(testament)
                })
        
        print(f"📖 Created {len(chunks)} book chunks (including splits for large books)")
        return chunks

class NeonUploader:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def create_tables(self):
        """Create the necessary tables in Neon."""
        with psycopg2.connect(
            self.connection_string, 
            sslmode='require', 
            connect_timeout=30,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        ) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create chapter_chunks table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chapter_chunks (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(768),
                        book VARCHAR(100),
                        chapter INTEGER,
                        verse_count INTEGER,
                        verse_references JSONB,
                        testament VARCHAR(10),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Create book_chunks table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS book_chunks (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(768),
                        book VARCHAR(100),
                        chapter_count INTEGER,
                        verse_count INTEGER,
                        chunk_type VARCHAR(20) DEFAULT 'book',
                        is_split BOOLEAN DEFAULT FALSE,
                        original_book VARCHAR(100),
                        testament VARCHAR(10),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                
                # Add new columns if they don't exist (for existing tables)
                conn.rollback()  # Reset any failed transaction
                
                def add_column_if_not_exists(table, column, definition):
                    try:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition};")
                        conn.commit()
                    except psycopg2.errors.DuplicateColumn:
                        conn.rollback()  # Column already exists
                    except Exception as e:
                        conn.rollback()
                        print(f"Note: Column {column} may already exist: {e}")
                
                add_column_if_not_exists("chapter_chunks", "testament", "VARCHAR(10)")
                add_column_if_not_exists("book_chunks", "chunk_type", "VARCHAR(20) DEFAULT 'book'")
                add_column_if_not_exists("book_chunks", "is_split", "BOOLEAN DEFAULT FALSE")
                add_column_if_not_exists("book_chunks", "original_book", "VARCHAR(100)")
                add_column_if_not_exists("book_chunks", "testament", "VARCHAR(10)")
                
                # Create indexes for similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chapter_chunks_embedding_idx 
                    ON chapter_chunks USING ivfflat (embedding vector_cosine_ops);
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS book_chunks_embedding_idx 
                    ON book_chunks USING ivfflat (embedding vector_cosine_ops);
                """)
                
                conn.commit()
                print("✅ Tables created successfully")
    
    def generate_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding using Google's text-embedding-004 model with retry logic."""
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model="text-embedding-004",
                    content=text
                )
                
                # Handle the response structure
                embedding = None
                if hasattr(result, 'embedding'):
                    embedding = result.embedding
                elif hasattr(result, 'embeddings') and result.embeddings:
                    embedding = result.embeddings[0].values
                else:
                    embedding = result['embedding']
                
                return embedding
            except Exception as e:
                if ("429" in str(e) or "504" in str(e)) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                    error_type = "Rate limited" if "429" in str(e) else "Timeout"
                    print(f"⚠️  {error_type} (attempt {attempt + 1}/{max_retries}). Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Error generating embedding: {e}")
                    return []
    
    def upload_chapter_chunks(self, chunks: List[Dict[str, Any]]):
        """Upload chapter chunks with embeddings to Neon."""
        print(f"📤 Uploading {len(chunks)} chapter chunks...")
        
        with psycopg2.connect(
            self.connection_string, 
            sslmode='require', 
            connect_timeout=30,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        ) as conn:
            with conn.cursor() as cur:
                # Clear existing chapter chunks
                cur.execute("DELETE FROM chapter_chunks;")
                print("🗑️  Cleared existing chapter chunks")
                
                for i, chunk in enumerate(chunks):
                    print(f"📝 Processing chapter chunk {i+1}/{len(chunks)}: {chunk['book']} {chunk['chapter']}")
                    
                    # Generate embedding
                    embedding = self.generate_embedding(chunk['content'])
                    if not embedding:
                        print(f"⚠️  Skipping chunk {i+1} due to embedding error")
                        continue
                    
                    # Insert into database
                    cur.execute("""
                        INSERT INTO chapter_chunks 
                        (content, embedding, book, chapter, verse_count, verse_references, testament)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        chunk['content'],
                        embedding,
                        str(chunk['book']),
                        int(chunk['chapter']),
                        int(chunk['verse_count']),
                        json.dumps(chunk['verse_references']),
                        chunk['testament']
                    ))
                    
                    # Debug: Print testament for key books
                    if chunk['book'] in ['Genesis', 'Psalms', 'Matthew', 'Revelation']:
                        print(f"  📖 {chunk['book']} {chunk['chapter']}: testament = {chunk['testament']}")
                    
                    # Commit every 10 chunks to avoid long transactions
                    if (i + 1) % 10 == 0:
                        conn.commit()
                        print(f"✅ Committed {i+1} chapter chunks")
                
                conn.commit()
                print(f"✅ Successfully uploaded {len(chunks)} chapter chunks")
    
    def upload_book_chunks(self, chunks: List[Dict[str, Any]]):
        """Upload book chunks with embeddings to Neon."""
        print(f"📤 Uploading {len(chunks)} book chunks...")
        
        with psycopg2.connect(
            self.connection_string, 
            sslmode='require', 
            connect_timeout=30,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5
        ) as conn:
            with conn.cursor() as cur:
                # Clear existing book chunks
                cur.execute("DELETE FROM book_chunks;")
                print("🗑️  Cleared existing book chunks")
                
                for i, chunk in enumerate(chunks):
                    print(f"📝 Processing book chunk {i+1}/{len(chunks)}: {chunk['book']}")
                    
                    # Generate embedding
                    embedding = self.generate_embedding(chunk['content'])
                    if not embedding:
                        print(f"⚠️  Skipping chunk {i+1} due to embedding error")
                        continue
                    
                    # Insert into database
                    cur.execute("""
                        INSERT INTO book_chunks 
                        (content, embedding, book, chapter_count, verse_count, chunk_type, is_split, original_book, testament)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        chunk['content'],
                        embedding,
                        str(chunk['book']),
                        int(chunk['chapter_count']),
                        int(chunk['verse_count']),
                        str(chunk['chunk_type']),
                        bool(chunk['is_split']),
                        str(chunk.get('original_book', '')) if chunk.get('original_book') else None,
                        str(chunk['testament'])
                    ))
                    
                    # Debug: Print testament for key books
                    if chunk['book'] in ['Genesis', 'Psalms', 'Matthew', 'Revelation']:
                        print(f"  📖 {chunk['book']}: testament = {chunk['testament']}")
                    
                    # Commit every 5 chunks to avoid long transactions
                    if (i + 1) % 5 == 0:
                        conn.commit()
                        print(f"✅ Committed {i+1} book chunks")
                
                conn.commit()
                print(f"✅ Successfully uploaded {len(chunks)} book chunks")

def main():
    print("📖 Bible Embeddings Generator")
    print("=" * 40)
    
    # Configuration
    csv_path = "data/bible_berean_translation - bible.csv"
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        return
    
    # Initialize chunker and uploader
    chunker = BibleChunker(csv_path)
    uploader = NeonUploader(database_url)
    
    # Create tables
    print("\n🏗️  Creating database tables...")
    uploader.create_tables()
    
    # Generate chapter chunks
    print("\n📚 Generating chapter chunks...")
    chapter_chunks = chunker.chunk_by_chapter()
    
    # Upload chapter chunks
    print("\n📤 Uploading chapter chunks...")
    uploader.upload_chapter_chunks(chapter_chunks)
    
    # Generate book chunks
    print("\n📖 Generating book chunks...")
    book_chunks = chunker.chunk_by_book()
    
    # Upload book chunks
    print("\n📤 Uploading book chunks...")
    uploader.upload_book_chunks(book_chunks)
    
    print("\n🎉 All done!")
    print(f"📊 Summary:")
    print(f"   - Chapter chunks: {len(chapter_chunks)}")
    print(f"   - Book chunks: {len(book_chunks)}")
    print(f"   - Total verses processed: {len(chunker.df)}")

if __name__ == "__main__":
    main() 