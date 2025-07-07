#!/usr/bin/env python3
"""
Test script to verify database connection and Bible data structure.
"""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test basic connection to Neon database."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        return False
    
    try:
        print("🔌 Testing connection to Neon...")
        with psycopg2.connect(database_url, sslmode='require') as conn:
            with conn.cursor() as cur:
                # Test basic query
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"✅ Connected successfully!")
                print(f"   PostgreSQL version: {version}")
                
                # Test pgvector extension
                cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                if cur.fetchone():
                    print("✅ pgvector extension is installed")
                else:
                    print("⚠️  pgvector extension not found - will be created by main script")
                
                return True
                
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_csv_data():
    """Test the CSV data structure."""
    csv_path = "data/bible_berean_translation - bible.csv"
    
    try:
        print(f"\n📖 Testing CSV data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        print(f"✅ CSV loaded successfully!")
        print(f"   Total verses: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check unique books and chapters
        unique_books = df['book'].unique()
        print(f"   Unique books: {len(unique_books)}")
        print(f"   Sample books: {unique_books[:5].tolist()}")
        
        # Check chapter distribution
        chapter_counts = df.groupby(['book', 'chapter']).size()
        print(f"   Total chapters: {len(chapter_counts)}")
        print(f"   Average verses per chapter: {chapter_counts.mean():.1f}")
        
        # Show sample data
        print(f"\n📝 Sample data:")
        print(df.head(3).to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ CSV test failed: {e}")
        return False

def test_tables():
    """Test if the embedding tables exist."""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        return False
    
    try:
        with psycopg2.connect(database_url, sslmode='require') as conn:
            with conn.cursor() as cur:
                # Check if tables exist
                tables = ['chapter_chunks', 'book_chunks']
                
                for table in tables:
                    cur.execute(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = '{table}'
                        );
                    """)
                    exists = cur.fetchone()[0]
                    
                    if exists:
                        # Count existing chunks
                        cur.execute(f"SELECT COUNT(*) FROM {table};")
                        count = cur.fetchone()[0]
                        print(f"✅ {table} table exists with {count} chunks")
                    else:
                        print(f"ℹ️  {table} table doesn't exist yet")
                
                return True
                
    except Exception as e:
        print(f"❌ Table test failed: {e}")
        return False

def main():
    print("🧪 Testing Bible Embeddings Setup")
    print("=" * 40)
    
    # Test connection
    if not test_connection():
        return
    
    # Test CSV data
    if not test_csv_data():
        return
    
    # Test tables
    print(f"\n🗄️  Testing database tables...")
    test_tables()
    
    print("\n✅ All tests completed!")
    print("\nNext steps:")
    print("1. Run 'uv run python bible_embeddings.py' to generate and upload embeddings")
    print("2. The script will create two tables:")
    print("   - chapter_chunks: One chunk per chapter")
    print("   - book_chunks: One chunk per book")

if __name__ == "__main__":
    main() 