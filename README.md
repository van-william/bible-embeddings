# Bible Embeddings

This project creates embeddings for Bible text by chunking verses into chapters and books, then uploading them to a Neon database with vector search capabilities.

## Setup

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

   **Optional Dependencies** (install as needed):
   ```bash
   # Advanced analysis features
   uv add --optional analysis   # UMAP, HDBSCAN clustering
   
   # Web serving capabilities  
   uv add --optional web       # Flask, Jupyter
   
   # Development tools
   uv add --optional dev       # pytest, black, mypy
   ```

3. **Environment Variables**:
   Make sure your `.env` file contains:
   ```
   VITE_GEMINI_API_KEY=your_gemini_api_key
   DATABASE_URL=your_neon_database_url
   ```

## Usage

### 1. Test Setup
First, test your database connection and data:
```bash
uv run python test_bible_setup.py
```

### 2. Generate Embeddings
Run the main script to chunk and upload embeddings:
```bash
uv run python bible_embeddings.py
```

This will:
- Create two tables: `chapter_chunks` and `book_chunks`
- Generate embeddings for each chapter and book
- Upload them to your Neon database

### 3. Run Enhanced Analysis
Generate comprehensive visualizations and analysis:
```bash
uv run python clean_bible_analysis.py
```

This creates the `results/` directory with:
- **9 PNG visualizations** - Static charts and graphs
- **Interactive HTML** - Web-based analysis with zoom/pan
- **CSV/JSON exports** - Raw data for further analysis
- **Comprehensive report** - Markdown summary with insights

**Analysis Features:**
- 📊 Testament separation (OT vs NT)
- 📚 Book categories (Law, Wisdom, Prophets, etc.)
- 🎨 t-SNE clustering visualization
- 📖 Author groupings (Moses, Paul, John, etc.)
- 📝 Literary genres (Narrative, Poetry, Prophetic, etc.)
- 🕸️ Network analysis of book relationships
- 📈 Chapter progression within books
- 🌐 Interactive web visualization

### 4. Interactive Analysis (Optional)
For notebook-style exploration:
```bash
uv run marimo edit marimo_notebooks/bible_interactive_analysis.py
```

## Database Schema

### Chapter Chunks Table
- `id`: Primary key
- `content`: Full chapter text
- `embedding`: Vector embedding (768 dimensions)
- `book`: Book name
- `chapter`: Chapter number
- `verse_count`: Number of verses in chapter
- `verse_references`: JSON array of verse references
- `testament`: Testament ('OT' for Old Testament, 'NT' for New Testament)
- `created_at`: Timestamp

### Book Chunks Table
- `id`: Primary key
- `content`: Full book text (or chapter text for split books)
- `embedding`: Vector embedding (768 dimensions)
- `book`: Book name
- `chapter_count`: Number of chapters in chunk
- `verse_count`: Total verses in chunk
- `chunk_type`: Type of chunk ('book' or 'book_chapter')
- `is_split`: Whether this chunk was split from a larger book
- `original_book`: Original book name (for split chunks)
- `testament`: Testament ('OT' for Old Testament, 'NT' for New Testament)
- `created_at`: Timestamp

## Data Source

The script processes `data/bible_berean_translation - bible.csv` which contains:
- One verse per row
- Columns: reference, text, chapter, book, testament

## Features

- **Chapter-level chunks**: Each chapter becomes one embedding
- **Book-level chunks**: Each book becomes one embedding
- **Automatic retry logic**: Handles API rate limits and timeouts
- **Progress tracking**: Shows progress during upload
- **Vector search ready**: Includes pgvector indexes for similarity search

## Project Structure

```
bible-embeddings/
├── bible_embeddings.py          # Main script to generate embeddings
├── clean_bible_analysis.py      # Enhanced analysis with visualizations
├── test_bible_setup.py          # Database connection test
├── pyproject.toml               # Dependencies and project config
├── data/                        # Source Bible CSV data
├── marimo_notebooks/            # Interactive analysis notebooks
└── results/                     # Generated analysis outputs
    ├── images/                  # PNG visualizations
    ├── reports/                 # Markdown analysis reports
    ├── data/                    # CSV/JSON exports
    └── interactive_bible_analysis.html
```

## Example Queries

After running the script, you can perform vector similarity searches:

```sql
-- Find similar chapters
SELECT book, chapter, content 
FROM chapter_chunks 
ORDER BY embedding <=> '[your_query_embedding]' 
LIMIT 5;

-- Find similar books
SELECT book, content 
FROM book_chunks 
ORDER BY embedding <=> '[your_query_embedding]' 
LIMIT 3;
```

## Dependencies

**Core dependencies** (installed automatically):
- pandas, numpy, matplotlib, seaborn, scikit-learn
- psycopg2-binary (PostgreSQL adapter)
- google-generativeai (embeddings)
- plotly (interactive charts)
- networkx (network analysis)
- marimo (notebooks)

**Optional dependencies** (install as needed):
- **analysis**: UMAP, HDBSCAN for advanced clustering
- **web**: Flask, Jupyter for web serving
- **dev**: pytest, black, mypy for development 

---

## Graph (Neo4j): curated prophecy fulfillments

Add fulfillment mappings in `data/bible_fulfillment.md` (one per line):
- `Micah 5:2 -> Matthew 2:1`
- `Zechariah 9:9 -> Matthew 21:4-5`

Build and load into Neo4j:
```bash
uv run python graph/ingest/build_fulfills.py
cp results/graph/csv/fulfills_edges.csv graph/import/csv/
docker cp graph/import/csv/fulfills_edges.csv bible_neo4j:/var/lib/neo4j/import/csv/
docker exec bible_neo4j cypher-shell -u neo4j -p password -f /var/lib/neo4j/import/load_csv.cypher | cat
```

Verify in Neo4j:
```bash
docker exec bible_neo4j cypher-shell -u neo4j -p password "MATCH ()-[r:FULFILLS]->() RETURN count(r);" | cat
```

For full graph setup, login, and using Neon embeddings for references, see `graph/README.md`.