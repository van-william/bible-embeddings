import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import psycopg2
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics.pairwise import cosine_similarity
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    return (
        TSNE,
        PCA,
        cosine_similarity,
        go,
        load_dotenv,
        make_subplots,
        mo,
        np,
        os,
        pd,
        plotly,
        psycopg2,
        px,
    )


@app.cell
def __(mo):
    mo.md("""
    # 📖 Interactive Bible Embeddings Analysis
    
    **Goal**: Explore how Bible books relate to each other through embeddings
    - **Book relationships**: Which books are most similar?
    - **OT vs NT analysis**: How distinct are the testaments?
    - **Convergence points**: Where do OT and NT themes overlap?
    
    **Data Source**: Neon Database (single source of truth)
    """)
    return


@app.cell
def __(mo, os, pd, psycopg2):
    # Load data from database
    def load_bible_data():
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            return None, None
        
        with psycopg2.connect(database_url, sslmode='require') as conn:
            # Load book chunks
            book_query = """
                SELECT id, book, testament, chunk_type, is_split, original_book, 
                       chapter_count, verse_count, embedding
                FROM book_chunks
                ORDER BY book, id
            """
            
            # Load chapter chunks  
            chapter_query = """
                SELECT id, book, testament, chapter, verse_count, embedding
                FROM chapter_chunks
                ORDER BY book, chapter
            """
            
            book_df = pd.read_sql_query(book_query, conn)
            chapter_df = pd.read_sql_query(chapter_query, conn)
        
        return book_df, chapter_df
    
    book_df, chapter_df = load_bible_data()
    
    if book_df is not None:
        mo.md(f"""
        ✅ **Data Loaded Successfully**
        - 📖 Book chunks: {len(book_df):,}
        - 📚 Chapter chunks: {len(chapter_df):,}
        - 🔗 Testament distribution: {book_df['testament'].value_counts().to_dict()}
        """)
    else:
        mo.md("❌ **Error**: Could not load data. Check DATABASE_URL environment variable.")
    return book_df, chapter_df, load_bible_data


@app.cell
def __(book_df, np):
    # Prepare embeddings
    def prepare_embeddings(df, embedding_col='embedding'):
        embeddings = []
        for emb_str in df[embedding_col]:
            if emb_str:
                emb_str = emb_str.strip('[]')
                emb_array = np.array([float(x.strip()) for x in emb_str.split(',')])
                embeddings.append(emb_array)
            else:
                embeddings.append(np.zeros(768))
        return np.array(embeddings)
    
    book_embeddings = prepare_embeddings(book_df) if book_df is not None else None
    return book_embeddings, prepare_embeddings


@app.cell
def __(book_df, book_embeddings, np, pd):
    # Aggregate books into canonical 66 books
    def aggregate_books(book_df, book_embeddings):
        aggregated_books = []
        aggregated_embeddings = []
        
        # Get unique books
        all_books = set()
        for _, row in book_df.iterrows():
            if row['is_split']:
                all_books.add(row['original_book'])
            else:
                all_books.add(row['book'])
        
        for book_name in sorted(all_books):
            # Get all chunks for this book
            book_chunks = book_df[
                ((book_df['book'] == book_name) & (~book_df['is_split'])) |
                (book_df['original_book'] == book_name)
            ]
            
            if len(book_chunks) == 0:
                continue
            
            # Get embeddings for this book
            chunk_indices = book_chunks.index
            book_chunk_embeddings = book_embeddings[chunk_indices]
            
            # Aggregate using mean
            mean_embedding = np.mean(book_chunk_embeddings, axis=0)
            
            # Get metadata
            first_chunk = book_chunks.iloc[0]
            testament = first_chunk['testament']
            total_verses = book_chunks['verse_count'].sum()
            
            aggregated_books.append({
                'book': book_name,
                'testament': testament,
                'verse_count': total_verses,
                'chunk_count': len(book_chunks)
            })
            aggregated_embeddings.append(mean_embedding)
        
        return pd.DataFrame(aggregated_books), np.array(aggregated_embeddings)
    
    if book_df is not None and book_embeddings is not None:
        aggregated_df, aggregated_embeddings = aggregate_books(book_df, book_embeddings)
    else:
        aggregated_df, aggregated_embeddings = None, None
    return aggregate_books, aggregated_df, aggregated_embeddings


@app.cell
def __(aggregated_df, mo):
    if aggregated_df is not None:
        mo.md(f"""
        📊 **Canonical Bible Books**: {len(aggregated_df)} books aggregated
        - **Old Testament**: {len(aggregated_df[aggregated_df['testament'] == 'OT'])} books
        - **New Testament**: {len(aggregated_df[aggregated_df['testament'] == 'NT'])} books
        """)
    return


@app.cell
def __(mo):
    # Interactive controls
    mo.md("## 🎛️ Interactive Controls")
    return


@app.cell
def __(aggregated_df, mo):
    # Book selection
    if aggregated_df is not None:
        book_options = sorted(aggregated_df['book'].tolist())
        selected_book = mo.ui.dropdown(
            options=book_options,
            value="Genesis",
            label="Select a book to analyze:"
        )
        selected_book
    else:
        selected_book = None
    return book_options, selected_book


@app.cell
def __(mo):
    # Analysis type selection
    analysis_type = mo.ui.radio(
        options=["Testament Separation", "Book Similarities", "Cross-Testament Convergence"],
        value="Testament Separation",
        label="Choose analysis type:"
    )
    analysis_type
    return analysis_type,


@app.cell
def __(mo):
    # Visualization method
    viz_method = mo.ui.radio(
        options=["PCA", "t-SNE"],
        value="PCA", 
        label="Dimensionality reduction method:"
    )
    viz_method
    return viz_method,


@app.cell
def __(PCA, TSNE, aggregated_embeddings, viz_method):
    # Apply dimensionality reduction based on selection
    if aggregated_embeddings is not None:
        if viz_method.value == "PCA":
            reducer = PCA(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(aggregated_embeddings)
            variance_explained = f"PC1: {reducer.explained_variance_ratio_[0]:.1%}, PC2: {reducer.explained_variance_ratio_[1]:.1%}"
        else:  # t-SNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(aggregated_embeddings)-1))
            coords_2d = reducer.fit_transform(aggregated_embeddings)
            variance_explained = "t-SNE (non-linear embedding)"
    else:
        coords_2d = None
        variance_explained = ""
    return coords_2d, reducer, variance_explained


@app.cell
def __(aggregated_df, analysis_type, coords_2d, mo, px, variance_explained):
    # Create main visualization
    if coords_2d is not None and analysis_type.value == "Testament Separation":
        # Add coordinates to dataframe
        plot_df = aggregated_df.copy()
        plot_df['x'] = coords_2d[:, 0]
        plot_df['y'] = coords_2d[:, 1]
        
        # Create interactive plot
        fig = px.scatter(
            plot_df, 
            x='x', y='y',
            color='testament',
            hover_data=['book', 'verse_count'],
            text='book',
            title=f"Bible Books: Old Testament vs New Testament ({variance_explained})",
            color_discrete_map={'OT': 'blue', 'NT': 'orange'},
            width=900, height=700
        )
        
        fig.update_traces(textposition="top center", textfont_size=10)
        fig.update_layout(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            showlegend=True
        )
        
        mo.ui.plotly(fig)
    else:
        mo.md("Select 'Testament Separation' to see the visualization")
    return fig, plot_df


@app.cell
def __(aggregated_df, aggregated_embeddings, analysis_type, cosine_similarity, mo):
    # Book similarity analysis
    if analysis_type.value == "Book Similarities" and aggregated_df is not None:
        def analyze_book_similarities(target_book, top_k=10):
            if target_book not in aggregated_df['book'].values:
                return None
                
            target_idx = aggregated_df[aggregated_df['book'] == target_book].index[0]
            target_embedding = aggregated_embeddings[target_idx:target_idx+1]
            
            # Calculate similarities to all books
            similarities = cosine_similarity(target_embedding, aggregated_embeddings)[0]
            
            # Get top similar books (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
            
            similar_books = []
            for idx in similar_indices:
                book = aggregated_df.iloc[idx]['book']
                testament = aggregated_df.iloc[idx]['testament']
                similarity = similarities[idx]
                similar_books.append({
                    'Book': book,
                    'Testament': testament, 
                    'Similarity': f"{similarity:.3f}"
                })
            
            return similar_books
        
        # Show analysis for selected book
        if hasattr(selected_book, 'value') and selected_book.value:
            similar_books = analyze_book_similarities(selected_book.value)
            if similar_books:
                similar_df = pd.DataFrame(similar_books)
                mo.md(f"## 📖 Books Most Similar to **{selected_book.value}**")
                mo.ui.table(similar_df)
            else:
                mo.md(f"Book '{selected_book.value}' not found")
        else:
            mo.md("Select a book above to see similarities")
    else:
        mo.md("Select 'Book Similarities' to see this analysis")
    return analyze_book_similarities, similar_books, similar_df


@app.cell
def __(aggregated_df, aggregated_embeddings, analysis_type, cosine_similarity, mo, pd):
    # Cross-testament convergence analysis
    if analysis_type.value == "Cross-Testament Convergence" and aggregated_df is not None:
        ot_books = aggregated_df[aggregated_df['testament'] == 'OT']
        nt_books = aggregated_df[aggregated_df['testament'] == 'NT']
        
        ot_embeddings = aggregated_embeddings[aggregated_df['testament'] == 'OT']
        nt_embeddings = aggregated_embeddings[aggregated_df['testament'] == 'NT']
        
        # Calculate cross-testament similarities
        cross_similarities = cosine_similarity(ot_embeddings, nt_embeddings)
        
        # Find top similarities
        top_pairs = []
        for i in range(len(ot_books)):
            for j in range(len(nt_books)):
                similarity = cross_similarities[i, j]
                ot_book = ot_books.iloc[i]['book']
                nt_book = nt_books.iloc[j]['book']
                top_pairs.append({
                    'OT Book': ot_book,
                    'NT Book': nt_book,
                    'Similarity': f"{similarity:.3f}"
                })
        
        # Sort by similarity and take top 15
        top_pairs_df = pd.DataFrame(top_pairs)
        top_pairs_df['Similarity_float'] = top_pairs_df['Similarity'].astype(float)
        top_pairs_df = top_pairs_df.sort_values('Similarity_float', ascending=False).head(15)
        top_pairs_df = top_pairs_df.drop('Similarity_float', axis=1)
        
        mo.md("## 🔄 Cross-Testament Convergence")
        mo.md("Books that show the highest thematic similarity across testaments:")
        mo.ui.table(top_pairs_df)
    else:
        mo.md("Select 'Cross-Testament Convergence' to see this analysis")
    return (
        cross_similarities,
        nt_books,
        nt_embeddings,
        ot_books,
        ot_embeddings,
        top_pairs,
        top_pairs_df,
    )


@app.cell
def __(aggregated_df, aggregated_embeddings, cosine_similarity, mo, np):
    # Testament separation metrics
    if aggregated_df is not None:
        ot_embeddings_sep = aggregated_embeddings[aggregated_df['testament'] == 'OT']
        nt_embeddings_sep = aggregated_embeddings[aggregated_df['testament'] == 'NT']
        
        # Calculate centroids
        ot_centroid = np.mean(ot_embeddings_sep, axis=0)
        nt_centroid = np.mean(nt_embeddings_sep, axis=0)
        
        # Calculate similarities
        testament_similarity = cosine_similarity([ot_centroid], [nt_centroid])[0][0]
        
        # Within-testament similarities
        ot_similarities = cosine_similarity(ot_embeddings_sep)
        nt_similarities = cosine_similarity(nt_embeddings_sep)
        
        ot_upper = ot_similarities[np.triu_indices_from(ot_similarities, k=1)]
        nt_upper = nt_similarities[np.triu_indices_from(nt_similarities, k=1)]
        
        ot_avg_similarity = np.mean(ot_upper)
        nt_avg_similarity = np.mean(nt_upper)
        
        mo.md(f"""
        ## 📊 Testament Separation Analysis
        
        - **OT ↔ NT similarity**: {testament_similarity:.3f}
        - **Average OT internal similarity**: {ot_avg_similarity:.3f}  
        - **Average NT internal similarity**: {nt_avg_similarity:.3f}
        
        **Interpretation**: 
        {"✅ Good separation - testaments are more distinct from each other than books within testaments" if testament_similarity < min(ot_avg_similarity, nt_avg_similarity) else "⚠️ Limited separation - testaments show significant overlap"}
        """)
    return (
        nt_avg_similarity,
        nt_centroid,
        nt_embeddings_sep,
        nt_similarities,
        nt_upper,
        ot_avg_similarity,
        ot_centroid,
        ot_embeddings_sep,
        ot_similarities,
        ot_upper,
        testament_similarity,
    )


@app.cell
def __(mo):
    mo.md("""
    ## 🎯 Key Insights
    
    This interactive analysis helps answer:
    
    1. **Book Relationships**: Use the book similarity analysis to explore which books share similar themes and language patterns
    
    2. **Testament Distinction**: The PCA/t-SNE visualization shows how well the Old and New Testaments cluster separately in semantic space
    
    3. **Convergence Points**: Cross-testament analysis reveals where OT and NT books share the most thematic similarity, potentially indicating prophetic fulfillment or shared theological concepts
    
    **Technical Notes**:
    - Embeddings: Google text-embedding-004 (768 dimensions)
    - Similarity: Cosine similarity metric
    - Data: Neon database as single source of truth
    """)
    return


if __name__ == "__main__":
    app.run() 