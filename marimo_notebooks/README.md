# 📓 Interactive Bible Analysis Notebooks

This directory contains Marimo notebooks for interactive exploration of Bible embeddings.

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   - Copy `.env` file from parent directory with `DATABASE_URL`

3. **Run the notebook**:
   ```bash
   marimo run bible_interactive_analysis.py
   ```

## 📊 Features

### Interactive Analysis Types:
- **Testament Separation**: Visualize how OT and NT books cluster in embedding space
- **Book Similarities**: Find books most similar to any selected book
- **Cross-Testament Convergence**: Discover OT-NT thematic overlaps

### Interactive Controls:
- **Book Selection**: Choose any of the 66 Bible books
- **Visualization Method**: Switch between PCA and t-SNE
- **Analysis Type**: Toggle between different analysis modes

### Key Insights:
1. **Book Relationships**: Explore thematic connections between books
2. **Testament Distinction**: Measure semantic separation between OT and NT
3. **Convergence Points**: Find where OT prophecy and NT fulfillment align

## 🎯 Research Questions

This notebook helps answer:
- How are Bible books related to each other?
- How distinct are Old Testament vs New Testament themes?
- Where do we see the strongest cross-testament similarities?
- Which books show unexpected thematic connections?

## 🔧 Technical Details

- **Data Source**: Neon PostgreSQL database (single source of truth)
- **Embeddings**: Google text-embedding-004 (768 dimensions)
- **Similarity**: Cosine similarity metric
- **Visualization**: Interactive Plotly charts
- **Dimensionality Reduction**: PCA and t-SNE options 