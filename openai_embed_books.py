import os
import pandas as pd
import tiktoken
import openai
import psycopg2
import numpy as np
from datetime import datetime, timezone
import json

# Config
CSV_PATH = 'data/bible_berean_translation - bible.csv'
RESULTS_DIR = 'results/openai_results'
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'openai_book_embeddings.csv')
MODEL = 'text-embedding-3-large'
TOKEN_LIMIT = 8192
VECTOR_SIZE = 3072  # Updated for text-embedding-3-large
VERSION = 'v1'

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
db_url = os.getenv('DATABASE_URL')
if not openai.api_key:
    raise RuntimeError('OPENAI_API_KEY not set in environment')
if not db_url:
    raise RuntimeError('DATABASE_URL not set in environment')

# Table creation SQL
CREATE_TABLE_SQL = f'''
CREATE TABLE IF NOT EXISTS openai_book_embeddings (
    id SERIAL PRIMARY KEY,
    book VARCHAR(100) UNIQUE,
    embedding vector({VECTOR_SIZE}),
    token_count INTEGER,
    model VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    version VARCHAR(20),
    notes TEXT
);
'''

# Load data
print(f"📖 Loading Bible CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Get encoding for the model
try:
    enc = tiktoken.encoding_for_model(MODEL)
except Exception:
    enc = tiktoken.get_encoding('cl100k_base')

# Prepare books
book_rows = []
for book, group in df.groupby('book'):
    text = ' '.join(group['text'].astype(str).tolist())
    tokens = enc.encode(text)
    token_count = len(tokens)
    if token_count > TOKEN_LIMIT:
        print(f"⏩ Skipping {book} ({token_count} tokens > {TOKEN_LIMIT})")
        continue
    book_rows.append({'book': book, 'text': text, 'token_count': token_count})

print(f"\n✅ {len(book_rows)} books fit in a single chunk and will be embedded.")

# Connect to Neon and create table
with psycopg2.connect(db_url, sslmode='require') as conn:
    with conn.cursor() as cur:
        # Check if table exists and has correct dimensions
        cur.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'openai_book_embeddings' 
            AND column_name = 'embedding'
        """)
        result = cur.fetchone()
        
        if result:
            # Table exists, check if we need to recreate it
            cur.execute("DROP TABLE IF EXISTS openai_book_embeddings")
            print("🔄 Recreating table with correct dimensions...")
        
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        print("✅ Database table ready")

# Embed and store
embeddings = []
cost_per_1k = 0.00013  # USD
input_tokens_total = 0
for i, row in enumerate(book_rows):
    print(f"\n🔎 [{i+1}/{len(book_rows)}] Embedding {row['book']} ({row['token_count']} tokens)...")
    try:
        response = openai.embeddings.create(
            input=row['text'],
            model=MODEL
        )
        emb = response.data[0].embedding
        input_tokens = response.usage.total_tokens if hasattr(response, 'usage') else row['token_count']
        input_tokens_total += input_tokens
    except Exception as e:
        print(f"❌ Error embedding {row['book']}: {e}")
        continue
    # Store in DB (overwrite if exists)
    with psycopg2.connect(db_url, sslmode='require') as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO openai_book_embeddings (book, embedding, token_count, model, created_at, version, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (book) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    token_count = EXCLUDED.token_count,
                    model = EXCLUDED.model,
                    created_at = EXCLUDED.created_at,
                    version = EXCLUDED.version,
                    notes = EXCLUDED.notes;
            """,
            (row['book'], emb, row['token_count'], MODEL, datetime.now(timezone.utc), VERSION, None))
            conn.commit()
    # Save for CSV
    embeddings.append({
        'book': row['book'],
        'token_count': row['token_count'],
        'model': MODEL,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'version': VERSION,
        'embedding': json.dumps(emb)
    })
    print(f"✅ Embedded and stored {row['book']}")

# Save to CSV
out_df = pd.DataFrame(embeddings)
out_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Embeddings saved to {OUTPUT_CSV}")

# Print cost estimate
est_cost = (input_tokens_total / 1000) * cost_per_1k
print(f"\nTotal input tokens: {input_tokens_total}")
print(f"Estimated embedding cost (@ $0.00013/1k tokens): ${est_cost:.4f}") 