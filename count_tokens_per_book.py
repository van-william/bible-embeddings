import os
import pandas as pd
import tiktoken

# Config
CSV_PATH = 'data/bible_berean_translation - bible.csv'
RESULTS_DIR = 'results/openai_results'
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'book_token_counts.csv')
MODEL = 'text-embedding-3-large'
TOKEN_LIMIT = 8192

# Ensure output directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
print(f"📖 Loading Bible CSV from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

# Get encoding for the model
try:
    enc = tiktoken.encoding_for_model(MODEL)
except Exception:
    enc = tiktoken.get_encoding('cl100k_base')  # Fallback

# Count tokens per book
book_tokens = []
for book, group in df.groupby('book'):
    text = ' '.join(group['text'].astype(str).tolist())
    tokens = enc.encode(text)
    token_count = len(tokens)
    book_tokens.append({'book': book, 'token_count': token_count})
    if token_count > TOKEN_LIMIT:
        print(f"⚠️  {book} exceeds {TOKEN_LIMIT} tokens: {token_count} tokens")

# Save to CSV
out_df = pd.DataFrame(book_tokens)
out_df = out_df.sort_values('book')
out_df.to_csv(OUTPUT_CSV, index=False)

# Print summary
print(f"\n✅ Token counts per book saved to {OUTPUT_CSV}")
print("\nTop 5 largest books by token count:")
print(out_df.sort_values('token_count', ascending=False).head(5))

total_tokens = out_df['token_count'].sum()
cost_per_1k = 0.00013  # USD
est_cost = (total_tokens / 1000) * cost_per_1k
print(f"\nTotal tokens (all books): {total_tokens}")
print(f"Estimated embedding cost (@ $0.00013/1k tokens): ${est_cost:.4f}")

if (out_df['token_count'] > TOKEN_LIMIT).any():
    print("\n⚠️  Some books exceed the 8192 token limit and will need to be chunked for embedding.")
else:
    print("\n✅ All books fit within the 8192 token limit for a single OpenAI embedding call.") 