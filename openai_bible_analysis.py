#!/usr/bin/env python3
"""
OpenAI Bible Embeddings Analysis Script

Data Flow: OpenAI Embeddings → Analysis → Reports
Focus: Book relationships, OT vs NT analysis, similarity patterns
Enhanced: Multiple visualizations, book categories, separate PNG files
"""

import os
import pandas as pd
import numpy as np
import psycopg2
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

class OpenAIBibleAnalyzer:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise RuntimeError('DATABASE_URL not set in environment')
        
        # Define book categories
        self.book_categories = {
            'Law (Torah)': ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
            'Historical': ['Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
                          '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther'],
            'Wisdom': ['Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon'],
            'Major Prophets': ['Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel'],
            'Minor Prophets': ['Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 
                              'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi'],
            'Gospels': ['Matthew', 'Mark', 'Luke', 'John'],
            'Early Church': ['Acts'],
            "Paul's Epistles": ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 
                               'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', 
                               '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon'],
            'General Epistles': ['Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude'],
            'Apocalyptic': ['Revelation']
        }
        
        # Define author groupings
        self.author_groupings = {
            'Moses': ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
            'David': ['Psalms'],
            'Solomon': ['Proverbs', 'Ecclesiastes', 'Song of Solomon'],
            'Isaiah': ['Isaiah'],
            'Jeremiah': ['Jeremiah', 'Lamentations'],
            'Ezekiel': ['Ezekiel'],
            'Daniel': ['Daniel'],
            'Matthew': ['Matthew'],
            'Mark': ['Mark'],
            'Luke': ['Luke', 'Acts'],
            'John': ['John', '1 John', '2 John', '3 John', 'Revelation'],
            'Paul': ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 
                     'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', 
                     '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon'],
            'Peter': ['1 Peter', '2 Peter'],
            'James': ['James'],
            'Jude': ['Jude'],
            'Unknown/Other': ['Job', 'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', 
                             '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 
                             'Ezra', 'Nehemiah', 'Esther', 'Hosea', 'Joel', 'Amos', 
                             'Obadiah', 'Jonah', 'Micah', 'Nahum', 'Habakkuk', 
                             'Zephaniah', 'Haggai', 'Zechariah', 'Malachi', 'Hebrews']
        }
        
        # Define literary genres
        self.literary_genres = {
            'Legal': ['Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
            'Narrative': ['Genesis', 'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
                         '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Matthew', 'Mark', 'Luke', 'John', 'Acts'],
            'Poetry': ['Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon', 'Lamentations'],
            'Prophetic': ['Isaiah', 'Jeremiah', 'Ezekiel', 'Daniel'] + 
                        ['Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 
                         'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi'],
            'Epistolary': ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 
                          'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', 
                          '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon',
                          'Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude'],
            'Apocalyptic': ['Daniel', 'Revelation']
        }
        
    def create_output_directory(self):
        """Create organized output directory structure."""
        import os
        
        # Create main results directory
        os.makedirs('results/openai_results', exist_ok=True)
        os.makedirs('results/openai_results/images', exist_ok=True)
        os.makedirs('results/openai_results/reports', exist_ok=True)
        os.makedirs('results/openai_results/data', exist_ok=True)
        
        print("📁 Created organized output directories:")
        print("   - results/openai_results/images/    (PNG visualizations)")
        print("   - results/openai_results/reports/   (Markdown reports)")
        print("   - results/openai_results/data/      (CSV exports, JSON data)")
    
    def load_data_from_db(self):
        """Load OpenAI embeddings from database."""
        print("📊 Loading OpenAI embeddings from database...")
        
        with psycopg2.connect(self.db_url, sslmode='require') as conn:
            # Load book embeddings
            query = """
            SELECT book, embedding, token_count, model, created_at, version
            FROM openai_book_embeddings
            ORDER BY book
            """
            self.aggregated_df = pd.read_sql(query, conn)
            
            # Convert embeddings from string to numpy arrays
            self.aggregated_df['embedding'] = self.aggregated_df['embedding'].apply(
                lambda x: np.array(json.loads(x)) if isinstance(x, str) else x
            )
            
            # Add metadata
            self.aggregated_df['testament'] = self.aggregated_df['book'].apply(self.get_testament)
            self.aggregated_df['category'] = self.aggregated_df['book'].apply(self.get_category)
            self.aggregated_df['author'] = self.aggregated_df['book'].apply(self.get_author)
            self.aggregated_df['genre'] = self.aggregated_df['book'].apply(self.get_genre)
            
            # Create embedding matrix
            self.aggregated_embeddings = np.array(self.aggregated_df['embedding'].tolist())
            
        print(f"✅ Loaded {len(self.aggregated_df)} books with {self.aggregated_embeddings.shape[1]}-dimensional embeddings")
    
    def get_testament(self, book):
        """Get testament for a book."""
        ot_books = [book for books in self.book_categories.values() for book in books 
                   if book in ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy', 
                              'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
                              '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther',
                              'Job', 'Psalms', 'Proverbs', 'Ecclesiastes', 'Song of Solomon',
                              'Isaiah', 'Jeremiah', 'Lamentations', 'Ezekiel', 'Daniel',
                              'Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 
                              'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi']]
        return 'OT' if book in ot_books else 'NT'
    
    def get_category(self, book):
        """Get category for a book."""
        for category, books in self.book_categories.items():
            if book in books:
                return category
        return 'Other'
    
    def get_author(self, book):
        """Get author for a book."""
        for author, books in self.author_groupings.items():
            if book in books:
                return author
        return 'Unknown'
    
    def get_genre(self, book):
        """Get literary genre for a book."""
        for genre, books in self.literary_genres.items():
            if book in books:
                return genre
        return 'Other'
    
    def analyze_testament_separation(self):
        """Analyze separation between Old and New Testaments."""
        print("🔍 Analyzing testament separation...")
        
        ot_mask = self.aggregated_df['testament'] == 'OT'
        nt_mask = self.aggregated_df['testament'] == 'NT'
        
        ot_embeddings = self.aggregated_embeddings[ot_mask]
        nt_embeddings = self.aggregated_embeddings[nt_mask]
        
        # Calculate similarities
        testament_similarity = np.mean(cosine_similarity(ot_embeddings, nt_embeddings))
        ot_avg_similarity = np.mean(cosine_similarity(ot_embeddings))
        nt_avg_similarity = np.mean(cosine_similarity(nt_embeddings))
        
        return {
            'testament_similarity': testament_similarity,
            'ot_avg_similarity': ot_avg_similarity,
            'nt_avg_similarity': nt_avg_similarity,
            'ot_count': np.sum(ot_mask),
            'nt_count': np.sum(nt_mask)
        }
    
    def find_cross_testament_similarities(self):
        """Find top cross-testament book similarities."""
        print("🔄 Finding cross-testament similarities...")
        
        ot_books = self.aggregated_df[self.aggregated_df['testament'] == 'OT']['book'].tolist()
        nt_books = self.aggregated_df[self.aggregated_df['testament'] == 'NT']['book'].tolist()
        
        similarities = cosine_similarity(self.aggregated_embeddings)
        
        cross_pairs = []
        for i, ot_book in enumerate(ot_books):
            ot_idx = self.aggregated_df[self.aggregated_df['book'] == ot_book].index[0]
            for j, nt_book in enumerate(nt_books):
                nt_idx = self.aggregated_df[self.aggregated_df['book'] == nt_book].index[0]
                similarity = similarities[ot_idx, nt_idx]
                cross_pairs.append((similarity, ot_book, nt_book))
        
        return sorted(cross_pairs, reverse=True)[:10]
    
    def analyze_book_similarities(self):
        """Analyze similarities for each book."""
        print("📚 Analyzing individual book similarities...")
        
        similarities = cosine_similarity(self.aggregated_embeddings)
        book_similarities = {}
        
        for i, row in self.aggregated_df.iterrows():
            book = row['book']
            book_sims = []
            
            for j, other_row in self.aggregated_df.iterrows():
                if i != j:
                    similarity = similarities[i, j]
                    testament = other_row['testament']
                    category = other_row['category']
                    book_sims.append((other_row['book'], testament, category, similarity))
            
            # Sort by similarity and take top 5
            book_sims.sort(key=lambda x: x[3], reverse=True)
            book_similarities[book] = book_sims[:5]
        
        return book_similarities
    
    def create_testament_visualization(self):
        """Create PCA visualization of testament separation."""
        print("📊 Creating testament separation visualization...")
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        colors = {'OT': '#1f77b4', 'NT': '#ff7f0e'}
        
        for testament in ['OT', 'NT']:
            mask = self.aggregated_df['testament'] == testament
            ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                      c=colors[testament], s=100, alpha=0.7, label=testament)
            
            # Add book labels
            for idx in self.aggregated_df[mask].index:
                book = self.aggregated_df.loc[idx, 'book']
                ax.annotate(book, (book_pca[idx, 0], book_pca[idx, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('OpenAI Bible Embeddings: Testament Separation\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/01_testament_separation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_category_visualization(self):
        """Create PCA visualization of book categories."""
        print("📊 Creating category visualization...")
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.book_categories)))
        
        for i, (category, books) in enumerate(self.book_categories.items()):
            mask = self.aggregated_df['book'].isin(books)
            if mask.any():
                ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                          c=[colors[i]], s=100, alpha=0.7, label=category)
                
                # Add book labels
                for idx in self.aggregated_df[mask].index:
                    book = self.aggregated_df.loc[idx, 'book']
                    ax.annotate(book, (book_pca[idx, 0], book_pca[idx, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('OpenAI Bible Embeddings: Book Categories\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/02_book_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_tsne_visualization(self):
        """Create t-SNE visualization."""
        print("📊 Creating t-SNE visualization...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(self.aggregated_df)-1))
        book_tsne = tsne.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        colors = {'OT': '#1f77b4', 'NT': '#ff7f0e'}
        
        for testament in ['OT', 'NT']:
            mask = self.aggregated_df['testament'] == testament
            ax.scatter(book_tsne[mask, 0], book_tsne[mask, 1], 
                      c=colors[testament], s=100, alpha=0.7, label=testament)
            
            # Add book labels
            for idx in self.aggregated_df[mask].index:
                book = self.aggregated_df.loc[idx, 'book']
                ax.annotate(book, (book_tsne[idx, 0], book_tsne[idx, 1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('t-SNE 1', fontsize=14)
        ax.set_ylabel('t-SNE 2', fontsize=14)
        ax.set_title('OpenAI Bible Embeddings: Natural Clusters\n(t-SNE Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/03_tsne_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_similarity_heatmap(self):
        """Create similarity heatmap."""
        print("📊 Creating similarity heatmap...")
        
        similarities = cosine_similarity(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        im = ax.imshow(similarities, cmap='viridis', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(self.aggregated_df)))
        ax.set_yticks(range(len(self.aggregated_df)))
        ax.set_xticklabels(self.aggregated_df['book'], rotation=45, ha='right')
        ax.set_yticklabels(self.aggregated_df['book'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        ax.set_title('OpenAI Bible Embeddings: Book Similarity Matrix', 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/04_category_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_cross_testament_heatmap(self):
        """Create cross-testament similarity heatmap."""
        print("📊 Creating cross-testament heatmap...")
        
        ot_books = self.aggregated_df[self.aggregated_df['testament'] == 'OT']['book'].tolist()
        nt_books = self.aggregated_df[self.aggregated_df['testament'] == 'NT']['book'].tolist()
        
        similarities = cosine_similarity(self.aggregated_embeddings)
        
        # Create cross-testament similarity matrix
        cross_similarities = []
        for ot_book in ot_books:
            ot_idx = self.aggregated_df[self.aggregated_df['book'] == ot_book].index[0]
            row = []
            for nt_book in nt_books:
                nt_idx = self.aggregated_df[self.aggregated_df['book'] == nt_book].index[0]
                row.append(similarities[ot_idx, nt_idx])
            cross_similarities.append(row)
        
        cross_similarities = np.array(cross_similarities)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        im = ax.imshow(cross_similarities, cmap='viridis', aspect='auto')
        
        # Set ticks
        ax.set_xticks(range(len(nt_books)))
        ax.set_yticks(range(len(ot_books)))
        ax.set_xticklabels(nt_books, rotation=45, ha='right')
        ax.set_yticklabels(ot_books)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        ax.set_title('OpenAI Bible Embeddings: Cross-Testament Similarities\n(OT vs NT)', 
                    fontsize=18, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/05_cross_testament_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_author_analysis(self):
        """Create author-based analysis."""
        print("📊 Creating author analysis...")
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Get unique authors
        authors = self.aggregated_df['author'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(authors)))
        
        for i, author in enumerate(authors):
            mask = self.aggregated_df['author'] == author
            if mask.sum() > 0:
                ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                          c=[colors[i]], s=100, alpha=0.7, label=f"{author} ({mask.sum()})")
                
                # Add book labels
                for idx in self.aggregated_df[mask].index:
                    book = self.aggregated_df.loc[idx, 'book']
                    ax.annotate(book, (book_pca[idx, 0], book_pca[idx, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('OpenAI Bible Embeddings: Author Analysis\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/07_author_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_genre_analysis(self):
        """Create literary genre analysis."""
        print("📊 Creating genre analysis...")
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Get unique genres
        genres = self.aggregated_df['genre'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(genres)))
        
        for i, genre in enumerate(genres):
            mask = self.aggregated_df['genre'] == genre
            if mask.sum() > 0:
                ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                          c=[colors[i]], s=100, alpha=0.7, label=f"{genre} ({mask.sum()})")
                
                # Add book labels
                for idx in self.aggregated_df[mask].index:
                    book = self.aggregated_df.loc[idx, 'book']
                    ax.annotate(book, (book_pca[idx, 0], book_pca[idx, 1]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('OpenAI Bible Embeddings: Literary Genre Analysis\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/openai_results/images/08_genre_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_plotly_visualization(self):
        """Create interactive Plotly visualization for web viewing."""
        print("📊 Creating interactive Plotly visualization...")
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.io as pio
        except ImportError:
            print("⚠️  Plotly not installed. Skipping interactive visualization.")
            print("   💡 Install with: uv add plotly")
            return
        
        # Apply PCA
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        # Create interactive dataframe
        plot_df = self.aggregated_df.copy()
        plot_df['PC1'] = book_pca[:, 0]
        plot_df['PC2'] = book_pca[:, 1]
        plot_df['hover_text'] = plot_df['book'] + '<br>' + \
                               plot_df['testament'] + '<br>' + \
                               plot_df['category'] + '<br>' + \
                               'Tokens: ' + plot_df['token_count'].astype(str)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('By Testament', 'By Category', 'By Author', 'By Genre'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Testament plot
        for testament in plot_df['testament'].unique():
            mask = plot_df['testament'] == testament
            fig.add_trace(
                go.Scatter(
                    x=plot_df[mask]['PC1'],
                    y=plot_df[mask]['PC2'],
                    mode='markers+text',
                    text=plot_df[mask]['book'],
                    textposition="middle center",
                    textfont=dict(size=6),
                    hovertext=plot_df[mask]['hover_text'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    name=testament,
                    marker=dict(size=8, opacity=0.8),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Category plot
        for category in plot_df['category'].unique():
            mask = plot_df['category'] == category
            fig.add_trace(
                go.Scatter(
                    x=plot_df[mask]['PC1'],
                    y=plot_df[mask]['PC2'],
                    mode='markers+text',
                    text=plot_df[mask]['book'],
                    textposition="middle center",
                    textfont=dict(size=6),
                    hovertext=plot_df[mask]['hover_text'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    name=category,
                    marker=dict(size=8, opacity=0.8),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Author plot
        for author in plot_df['author'].unique():
            mask = plot_df['author'] == author
            fig.add_trace(
                go.Scatter(
                    x=plot_df[mask]['PC1'],
                    y=plot_df[mask]['PC2'],
                    mode='markers+text',
                    text=plot_df[mask]['book'],
                    textposition="middle center",
                    textfont=dict(size=6),
                    hovertext=plot_df[mask]['hover_text'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    name=author,
                    marker=dict(size=8, opacity=0.8),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Genre plot
        for genre in plot_df['genre'].unique():
            mask = plot_df['genre'] == genre
            fig.add_trace(
                go.Scatter(
                    x=plot_df[mask]['PC1'],
                    y=plot_df[mask]['PC2'],
                    mode='markers+text',
                    text=plot_df[mask]['book'],
                    textposition="middle center",
                    textfont=dict(size=6),
                    hovertext=plot_df[mask]['hover_text'],
                    hovertemplate='%{hovertext}<extra></extra>',
                    name=genre,
                    marker=dict(size=8, opacity=0.8),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Interactive OpenAI Bible Analysis (PCA Projection)',
            title_font_size=20,
            height=800,
            showlegend=False
        )
        
        # Save as HTML
        fig.write_html('results/openai_results/interactive_bible_analysis.html')
        print("✅ Interactive visualization saved as 'results/openai_results/interactive_bible_analysis.html'")
    
    def export_data(self):
        """Export analysis data as CSV/JSON for further use."""
        print("📊 Exporting analysis data...")
        
        # Export book data
        self.aggregated_df.to_csv('results/openai_results/data/book_analysis.csv', index=False)
        
        # Export similarity matrices
        book_similarities = cosine_similarity(self.aggregated_embeddings)
        similarity_df = pd.DataFrame(
            book_similarities,
            index=self.aggregated_df['book'],
            columns=self.aggregated_df['book']
        )
        similarity_df.to_csv('results/openai_results/data/book_similarities.csv')
        
        # Export book categories as JSON
        categories_data = {
            'book_categories': self.book_categories,
            'author_groupings': self.author_groupings,
            'literary_genres': self.literary_genres
        }
        with open('results/openai_results/data/book_categories.json', 'w') as f:
            json.dump(categories_data, f, indent=2)
        
        print("✅ Data exported to results/openai_results/data/")
    
    def write_enhanced_report(self, testament_results, cross_testament_pairs, book_similarities):
        """Write comprehensive analysis report with PNG references."""
        print("📝 Writing enhanced analysis report...")
        
        with open('results/openai_results/reports/bible_enhanced_analysis.md', 'w') as f:
            f.write("# 📖 OpenAI Bible Embeddings Analysis\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("**Data Source**: OpenAI Embeddings (text-embedding-3-large)\n\n")
            
            # Dataset Overview
            f.write("## 📊 Dataset Overview\n\n")
            f.write(f"- **Total books analyzed**: {len(self.aggregated_df)}\n")
            f.write(f"- **Old Testament books**: {testament_results['ot_count']}\n")
            f.write(f"- **New Testament books**: {testament_results['nt_count']}\n")
            f.write(f"- **Embedding dimensions**: {self.aggregated_embeddings.shape[1]}\n")
            f.write(f"- **Total tokens**: {self.aggregated_df['token_count'].sum():,}\n\n")
            
            # Visualizations
            f.write("## 🎨 Visualizations\n\n")
            
            f.write("### 1. Testament Separation Analysis\n")
            f.write("![OT vs NT Separation](../images/01_testament_separation.png)\n\n")
            f.write("**Analysis**: PCA projection showing how well Old and New Testament books separate in embedding space.\n\n")
            
            f.write("### 2. Biblical Book Categories\n")
            f.write("![Book Categories](../images/02_book_categories.png)\n\n")
            f.write("**Analysis**: Books grouped by literary and theological categories (Law, Wisdom, Prophets, Gospels, Epistles, etc.).\n\n")
            
            f.write("### 3. Raw 2D Clusters (t-SNE)\n")
            f.write("![t-SNE Clusters](../images/03_tsne_clusters.png)\n\n")
            f.write("**Analysis**: t-SNE projection revealing natural clusters and relationships between books.\n\n")
            
            f.write("### 4. Category Similarity Matrix\n")
            f.write("![Category Similarities](../images/04_category_similarity_heatmap.png)\n\n")
            f.write("**Analysis**: Heatmap showing average similarities between different biblical book categories.\n\n")
            
            f.write("### 5. Cross-Testament Convergence\n")
            f.write("![Cross-Testament Heatmap](../images/05_cross_testament_heatmap.png)\n\n")
            f.write("**Analysis**: Detailed heatmap showing similarities between every OT and NT book pair.\n\n")
            
            f.write("### 6. Author Analysis\n")
            f.write("![Author Analysis](../images/07_author_analysis.png)\n\n")
            f.write("**Analysis**: Books grouped by traditional author attribution (Moses, Paul, John, etc.).\n\n")
            
            f.write("### 7. Literary Genre Analysis\n")
            f.write("![Genre Analysis](../images/08_genre_analysis.png)\n\n")
            f.write("**Analysis**: Books categorized by literary genre (Narrative, Poetry, Prophetic, Legal, Epistolary, Apocalyptic).\n\n")
            
            f.write("### 8. Interactive Visualization\n")
            f.write("🌐 **[Open Interactive Analysis](../interactive_bible_analysis.html)** (requires modern web browser)\n\n")
            f.write("**Features**: Zoom, pan, hover details, and multiple perspective views in one interactive interface.\n\n")
            
            # Testament Analysis
            f.write("## 🔍 Testament Analysis\n\n")
            f.write("### Quantitative Separation Analysis\n")
            f.write(f"- **OT ↔ NT similarity**: {testament_results['testament_similarity']:.3f}\n")
            f.write(f"- **Average OT internal similarity**: {testament_results['ot_avg_similarity']:.3f}\n")
            f.write(f"- **Average NT internal similarity**: {testament_results['nt_avg_similarity']:.3f}\n\n")
            
            f.write("### 📈 Interpretation\n")
            if testament_results['testament_similarity'] < testament_results['ot_avg_similarity']:
                f.write("✅ **Excellent separation**: Old and New Testaments are more semantically distinct from each other than books within each testament. This suggests clear thematic boundaries between the testaments.\n\n")
            else:
                f.write("⚠️ **Limited separation**: Old and New Testaments show significant thematic overlap. This could indicate strong continuity in themes across testaments.\n\n")
            
            # Cross-Testament Analysis
            f.write("## 🔄 Top Cross-Testament Similarities\n\n")
            f.write("These book pairs show the highest thematic similarity across testaments:\n\n")
            
            for i, (similarity, ot_book, nt_book) in enumerate(cross_testament_pairs, 1):
                f.write(f"{i}. **{ot_book}** ↔ **{nt_book}**: `{similarity:.3f}`\n")
            f.write("\n")
            
            # Book Similarity Analysis
            f.write("## 📚 Individual Book Analysis\n\n")
            
            for target_book, similar_books in book_similarities.items():
                f.write(f"### 📖 {target_book}\n")
                f.write("Most semantically similar books:\n")
                for i, (book, testament, category, similarity) in enumerate(similar_books, 1):
                    f.write(f"{i}. **{book}** ({testament}, {category}): `{similarity:.3f}`\n")
                f.write("\n")
            
            # Category Breakdown
            f.write("## 📋 Book Categories\n\n")
            for category, books in self.book_categories.items():
                f.write(f"**{category}**: {', '.join(books)}\n\n")
            
            # Data Files
            f.write("## 💾 Data Files\n\n")
            f.write("Analysis data is available in structured formats:\n")
            f.write("- `../data/book_analysis.csv` - Book metadata and categories\n")
            f.write("- `../data/book_similarities.csv` - Similarity matrix\n")
            f.write("- `../data/book_categories.json` - Category definitions\n\n")
            
            # Technical Details
            f.write("## 🔧 Technical Details\n\n")
            f.write("### Methodology\n")
            f.write("- **Embedding Model**: OpenAI text-embedding-3-large (3072 dimensions)\n")
            f.write("- **Dimensionality Reduction**: PCA and t-SNE for visualization\n")
            f.write("- **Similarity Metric**: Cosine similarity\n")
            f.write("- **Clustering**: Natural groupings based on embedding similarities\n")
            f.write("- **Categories**: Traditional biblical literary/theological classifications\n\n")
            
            f.write("### File Structure\n")
            f.write("```\n")
            f.write("results/openai_results/\n")
            f.write("├── images/          # PNG visualizations\n")
            f.write("├── reports/         # Analysis reports\n")
            f.write("└── data/           # Raw data exports\n")
            f.write("```\n\n")
            
            f.write("---\n")
            f.write("*This enhanced analysis was generated using OpenAI embeddings-based semantic analysis with multiple visualization techniques.*\n")
        
        print("✅ Enhanced report written to `results/openai_results/reports/bible_enhanced_analysis.md`")
    
    def run_complete_analysis(self):
        """Run the complete enhanced analysis pipeline."""
        print("🚀 Starting OpenAI Bible Analysis")
        print("=" * 50)
        
        # Create output directory
        self.create_output_directory()
        
        # Load data
        self.load_data_from_db()
        
        # Analyze testament separation
        testament_results = self.analyze_testament_separation()
        
        # Find cross-testament similarities
        cross_testament_pairs = self.find_cross_testament_similarities()
        
        # Create all visualizations
        print("\n📊 Creating visualizations...")
        self.create_testament_visualization()
        self.create_category_visualization()
        self.create_tsne_visualization()
        self.create_similarity_heatmap()
        self.create_cross_testament_heatmap()
        self.create_author_analysis()
        self.create_genre_analysis()
        self.create_interactive_plotly_visualization()
        
        # Export data
        self.export_data()
        
        # Analyze book similarities
        book_similarities = self.analyze_book_similarities()
        
        # Write enhanced report
        self.write_enhanced_report(testament_results, cross_testament_pairs, book_similarities)
        
        print("\n🎉 OpenAI analysis complete!")
        print("📁 Check the `results/openai_results/` directory:")
        print("   - `images/` - 7 PNG visualizations")
        print("   - `interactive_bible_analysis.html` - Interactive web visualization")
        print("   - `reports/` - Comprehensive markdown report")
        print("   - `data/` - CSV/JSON exports for further analysis")

def main():
    analyzer = OpenAIBibleAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 