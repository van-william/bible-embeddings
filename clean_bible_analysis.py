#!/usr/bin/env python3
"""
Clean Bible Embeddings Analysis Script

Data Flow: CSV → Neon DB → Analysis (single source of truth)
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

class BibleAnalyzer:
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        self.book_df = None
        self.chapter_df = None
        self.book_embeddings = None
        self.chapter_embeddings = None
        
        # Define biblical book categories
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
            'Paul\'s Epistles': ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 
                               'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', 
                               '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon'],
            'General Epistles': ['Hebrews', 'James', '1 Peter', '2 Peter', '1 John', '2 John', '3 John', 'Jude'],
            'Apocalyptic': ['Revelation']
        }
        
        # Define traditional author groupings
        self.author_groupings = {
            'Moses': ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy'],
            'David': ['Psalms'],  # Many psalms attributed to David
            'Solomon': ['Proverbs', 'Ecclesiastes', 'Song of Solomon'],
            'Isaiah': ['Isaiah'],
            'Jeremiah': ['Jeremiah', 'Lamentations'],
            'Ezekiel': ['Ezekiel'],
            'Daniel': ['Daniel'],
            'Paul': ['Romans', '1 Corinthians', '2 Corinthians', 'Galatians', 
                    'Ephesians', 'Philippians', 'Colossians', '1 Thessalonians', 
                    '2 Thessalonians', '1 Timothy', '2 Timothy', 'Titus', 'Philemon'],
            'Luke': ['Luke', 'Acts'],
            'John': ['John', '1 John', '2 John', '3 John', 'Revelation'],
            'Peter': ['1 Peter', '2 Peter'],
            'Unknown/Multiple': ['Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel', '1 Kings', '2 Kings',
                                '1 Chronicles', '2 Chronicles', 'Ezra', 'Nehemiah', 'Esther', 'Job',
                                'Matthew', 'Mark', 'Hebrews', 'James', 'Jude'] + 
                               ['Hosea', 'Joel', 'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum', 
                                'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi']
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
        os.makedirs('results', exist_ok=True)
        os.makedirs('results/images', exist_ok=True)
        os.makedirs('results/reports', exist_ok=True)
        os.makedirs('results/data', exist_ok=True)
        
        print("📁 Created organized output directories:")
        print("   - results/images/    (PNG visualizations)")
        print("   - results/reports/   (Markdown reports)")
        print("   - results/data/      (CSV exports, JSON data)")
        
    def load_data_from_db(self):
        """Load all data from Neon database (single source of truth)."""
        print("🔌 Loading data from Neon database...")
        
        with psycopg2.connect(self.database_url, sslmode='require') as conn:
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
            
            self.book_df = pd.read_sql_query(book_query, conn)
            self.chapter_df = pd.read_sql_query(chapter_query, conn)
        
        print(f"📖 Loaded {len(self.book_df)} book chunks")
        print(f"📚 Loaded {len(self.chapter_df)} chapter chunks")
        
        # Prepare embeddings
        self.book_embeddings = self._prepare_embeddings(self.book_df)
        self.chapter_embeddings = self._prepare_embeddings(self.chapter_df)
        
        print(f"📊 Book embeddings shape: {self.book_embeddings.shape}")
        print(f"📊 Chapter embeddings shape: {self.chapter_embeddings.shape}")
    
    def _prepare_embeddings(self, df, embedding_col='embedding'):
        """Convert embedding strings to numpy arrays."""
        embeddings = []
        for emb_str in df[embedding_col]:
            if emb_str:
                emb_str = emb_str.strip('[]')
                emb_array = np.array([float(x.strip()) for x in emb_str.split(',')])
                embeddings.append(emb_array)
            else:
                embeddings.append(np.zeros(768))
        return np.array(embeddings)
    
    def aggregate_books(self):
        """Aggregate split books back into complete 66 books."""
        print("📊 Aggregating books into canonical 66 books...")
        
        aggregated_books = []
        aggregated_embeddings = []
        
        # Get unique books (including handling splits)
        all_books = set()
        for _, row in self.book_df.iterrows():
            if row['is_split']:
                all_books.add(row['original_book'])
            else:
                all_books.add(row['book'])
        
        for book_name in sorted(all_books):
            # Get all chunks for this book
            book_chunks = self.book_df[
                ((self.book_df['book'] == book_name) & (~self.book_df['is_split'])) |
                (self.book_df['original_book'] == book_name)
            ]
            
            if len(book_chunks) == 0:
                continue
            
            # Get embeddings for this book
            chunk_indices = book_chunks.index
            book_chunk_embeddings = self.book_embeddings[chunk_indices]
            
            # Aggregate using mean
            mean_embedding = np.mean(book_chunk_embeddings, axis=0)
            
            # Get metadata
            first_chunk = book_chunks.iloc[0]
            testament = first_chunk['testament']
            total_verses = book_chunks['verse_count'].sum()
            
            # Determine category
            category = 'Other'
            for cat_name, books in self.book_categories.items():
                if book_name in books:
                    category = cat_name
                    break
            
            aggregated_books.append({
                'book': book_name,
                'testament': testament,
                'category': category,
                'verse_count': total_verses,
                'chunk_count': len(book_chunks)
            })
            aggregated_embeddings.append(mean_embedding)
        
        self.aggregated_df = pd.DataFrame(aggregated_books)
        self.aggregated_embeddings = np.array(aggregated_embeddings)
        
        print(f"📖 Aggregated into {len(self.aggregated_df)} canonical books")
        return self.aggregated_df, self.aggregated_embeddings

    def analyze_testament_separation(self):
        """Analyze how well OT and NT are separated in embedding space."""
        print("📊 Analyzing OT vs NT separation...")
        
        # Use aggregated books for clean analysis
        ot_books = self.aggregated_df[self.aggregated_df['testament'] == 'OT']
        nt_books = self.aggregated_df[self.aggregated_df['testament'] == 'NT']
        
        ot_embeddings = self.aggregated_embeddings[self.aggregated_df['testament'] == 'OT']
        nt_embeddings = self.aggregated_embeddings[self.aggregated_df['testament'] == 'NT']
        
        # Calculate centroids
        ot_centroid = np.mean(ot_embeddings, axis=0)
        nt_centroid = np.mean(nt_embeddings, axis=0)
        
        # Calculate similarity between testaments
        testament_similarity = cosine_similarity([ot_centroid], [nt_centroid])[0][0]
        
        # Calculate within-testament similarities
        ot_similarities = cosine_similarity(ot_embeddings)
        nt_similarities = cosine_similarity(nt_embeddings)
        
        # Get upper triangular values (excluding diagonal)
        ot_upper = ot_similarities[np.triu_indices_from(ot_similarities, k=1)]
        nt_upper = nt_similarities[np.triu_indices_from(nt_similarities, k=1)]
        
        results = {
            'testament_similarity': testament_similarity,
            'ot_avg_similarity': np.mean(ot_upper),
            'nt_avg_similarity': np.mean(nt_upper),
            'ot_count': len(ot_books),
            'nt_count': len(nt_books)
        }
        
        print(f"📊 OT vs NT similarity: {testament_similarity:.3f}")
        print(f"📊 Average OT internal similarity: {results['ot_avg_similarity']:.3f}")
        print(f"📊 Average NT internal similarity: {results['nt_avg_similarity']:.3f}")
        
        return results
    
    def find_cross_testament_similarities(self, top_k=10):
        """Find books that are most similar across testaments."""
        print("🔍 Finding cross-testament similarities...")
        
        ot_books = self.aggregated_df[self.aggregated_df['testament'] == 'OT']
        nt_books = self.aggregated_df[self.aggregated_df['testament'] == 'NT']
        
        ot_embeddings = self.aggregated_embeddings[self.aggregated_df['testament'] == 'OT']
        nt_embeddings = self.aggregated_embeddings[self.aggregated_df['testament'] == 'NT']
        
        # Calculate cross-testament similarities
        cross_similarities = cosine_similarity(ot_embeddings, nt_embeddings)
        
        # Find top similarities
        top_pairs = []
        for i in range(len(ot_books)):
            for j in range(len(nt_books)):
                similarity = cross_similarities[i, j]
                ot_book = ot_books.iloc[i]['book']
                nt_book = nt_books.iloc[j]['book']
                top_pairs.append((similarity, ot_book, nt_book))
        
        # Sort and return top pairs
        top_pairs.sort(reverse=True)
        
        print(f"🔍 Top {top_k} cross-testament similarities:")
        for i, (similarity, ot_book, nt_book) in enumerate(top_pairs[:top_k], 1):
            print(f"  {i}. {ot_book} ↔ {nt_book}: {similarity:.3f}")
        
        return top_pairs[:top_k]

    def create_testament_visualization(self):
        """Create PCA visualization showing OT vs NT separation."""
        print("📊 Creating OT vs NT visualization...")
        
        # Apply PCA
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Plot by testament
        ot_mask = self.aggregated_df['testament'] == 'OT'
        nt_mask = self.aggregated_df['testament'] == 'NT'
        
        ax.scatter(book_pca[ot_mask, 0], book_pca[ot_mask, 1], 
                  c='#2E86AB', s=120, alpha=0.8, label='Old Testament', edgecolors='white', linewidth=1)
        ax.scatter(book_pca[nt_mask, 0], book_pca[nt_mask, 1], 
                  c='#A23B72', s=120, alpha=0.8, label='New Testament', edgecolors='white', linewidth=1)
        
        # Add book labels
        for i, row in self.aggregated_df.iterrows():
            ax.annotate(row['book'], 
                       (book_pca[i, 0], book_pca[i, 1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('Bible Books: Old Testament vs New Testament\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/images/01_testament_separation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return pca
    
    def create_category_visualization(self):
        """Create visualization showing biblical book categories."""
        print("📊 Creating book categories visualization...")
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Create color palette for categories
        categories = self.aggregated_df['category'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))
        
        # Plot each category
        for category in categories:
            mask = self.aggregated_df['category'] == category
            ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                      c=[color_map[category]], s=150, alpha=0.8, 
                      label=category, edgecolors='white', linewidth=1)
        
        # Add book labels
        for i, row in self.aggregated_df.iterrows():
            ax.annotate(row['book'], 
                       (book_pca[i, 0], book_pca[i, 1]),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=8, fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('Bible Books by Literary/Theological Categories\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/images/02_book_categories.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_tsne_visualization(self):
        """Create t-SNE visualization showing raw clusters."""
        print("📊 Creating t-SNE raw clusters visualization...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=15, max_iter=1000)
        book_tsne = tsne.fit_transform(self.aggregated_embeddings)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Left plot: By Testament
        ot_mask = self.aggregated_df['testament'] == 'OT'
        nt_mask = self.aggregated_df['testament'] == 'NT'
        
        ax1.scatter(book_tsne[ot_mask, 0], book_tsne[ot_mask, 1], 
                   c='#2E86AB', s=120, alpha=0.8, label='Old Testament', edgecolors='white', linewidth=1)
        ax1.scatter(book_tsne[nt_mask, 0], book_tsne[nt_mask, 1], 
                   c='#A23B72', s=120, alpha=0.8, label='New Testament', edgecolors='white', linewidth=1)
        
        for i, row in self.aggregated_df.iterrows():
            ax1.annotate(row['book'], 
                        (book_tsne[i, 0], book_tsne[i, 1]),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=8, fontweight='bold', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax1.set_title('t-SNE Clusters by Testament', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right plot: By Category
        categories = self.aggregated_df['category'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        color_map = dict(zip(categories, colors))
        
        for category in categories:
            mask = self.aggregated_df['category'] == category
            ax2.scatter(book_tsne[mask, 0], book_tsne[mask, 1], 
                       c=[color_map[category]], s=120, alpha=0.8, 
                       label=category, edgecolors='white', linewidth=1)
        
        for i, row in self.aggregated_df.iterrows():
            ax2.annotate(row['book'], 
                        (book_tsne[i, 0], book_tsne[i, 1]),
                        xytext=(6, 6), textcoords='offset points',
                        fontsize=8, fontweight='bold', alpha=0.9,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax2.set_title('t-SNE Clusters by Category', fontsize=16, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Raw 2D Clusters of Bible Books (t-SNE)', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('results/images/03_tsne_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_similarity_heatmap(self):
        """Create similarity heatmap between book categories."""
        print("📊 Creating category similarity heatmap...")
        
        # Calculate similarities
        similarities = cosine_similarity(self.aggregated_embeddings)
        
        # Create category-level similarity matrix
        categories = sorted(self.aggregated_df['category'].unique())
        category_similarities = np.zeros((len(categories), len(categories)))
        
        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                mask1 = self.aggregated_df['category'] == cat1
                mask2 = self.aggregated_df['category'] == cat2
                
                # Get cross-category similarities
                cross_sims = similarities[np.ix_(mask1, mask2)]
                category_similarities[i, j] = np.mean(cross_sims)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(category_similarities, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_yticklabels(categories)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Cosine Similarity', fontsize=12)
        
        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(categories)):
                text = ax.text(j, i, f'{category_similarities[i, j]:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Inter-Category Similarity Matrix\n(Average Cosine Similarity between Book Categories)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig('results/images/04_category_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_cross_testament_heatmap(self):
        """Create detailed cross-testament similarity heatmap."""
        print("📊 Creating cross-testament convergence heatmap...")
        
        ot_books = self.aggregated_df[self.aggregated_df['testament'] == 'OT']
        nt_books = self.aggregated_df[self.aggregated_df['testament'] == 'NT']
        
        ot_embeddings = self.aggregated_embeddings[self.aggregated_df['testament'] == 'OT']
        nt_embeddings = self.aggregated_embeddings[self.aggregated_df['testament'] == 'NT']
        
        # Calculate cross-testament similarities
        cross_similarities = cosine_similarity(ot_embeddings, nt_embeddings)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 12))
        
        im = ax.imshow(cross_similarities, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(nt_books)))
        ax.set_yticks(range(len(ot_books)))
        ax.set_xticklabels(nt_books['book'].values, rotation=45, ha='right')
        ax.set_yticklabels(ot_books['book'].values)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Cosine Similarity', fontsize=12)
        
        ax.set_title('Cross-Testament Convergence\n(OT vs NT Book Similarities)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_xlabel('New Testament Books', fontsize=14)
        ax.set_ylabel('Old Testament Books', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('results/images/05_cross_testament_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_chapter_analysis(self):
        """Create chapter-level analysis - new feature!"""
        print("📊 Creating chapter-level analysis...")
        
        # Sample some interesting books for chapter analysis
        sample_books = ['Genesis', 'Psalms', 'Matthew', 'Romans', 'Revelation']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, book in enumerate(sample_books):
            if i >= len(axes):
                break
                
            # Get chapters for this book
            book_chapters = self.chapter_df[self.chapter_df['book'] == book]
            if len(book_chapters) < 2:
                continue
                
            # Get chapter embeddings
            chapter_indices = book_chapters.index
            chapter_embeddings = self.chapter_embeddings[chapter_indices]
            
            # Apply PCA
            if len(chapter_embeddings) > 1:
                pca = PCA(n_components=2)
                chapter_pca = pca.fit_transform(chapter_embeddings)
                
                # Plot chapters
                scatter = axes[i].scatter(chapter_pca[:, 0], chapter_pca[:, 1], 
                                        c=book_chapters['chapter'], cmap='viridis', 
                                        s=80, alpha=0.7, edgecolors='white', linewidth=1)
                
                # Add chapter labels
                for j, (_, chapter_row) in enumerate(book_chapters.iterrows()):
                    axes[i].annotate(f"Ch{chapter_row['chapter']}", 
                                   (chapter_pca[j, 0], chapter_pca[j, 1]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.8)
                
                axes[i].set_title(f'{book} Chapters\n({len(book_chapters)} chapters)', 
                                fontsize=14, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=axes[i], label='Chapter')
        
        # Remove empty subplots
        for j in range(len(sample_books), len(axes)):
            axes[j].remove()
        
        plt.suptitle('Chapter Progression Analysis\n(How chapters cluster within books)', 
                    fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('results/images/06_chapter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_author_analysis(self):
        """Create visualization showing traditional author groupings."""
        print("📊 Creating author groupings analysis...")
        
        # Add author information to aggregated data
        self.aggregated_df['author'] = 'Unknown'
        for author, books in self.author_groupings.items():
            mask = self.aggregated_df['book'].isin(books)
            self.aggregated_df.loc[mask, 'author'] = author
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Create color palette for authors
        authors = self.aggregated_df['author'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(authors)))
        color_map = dict(zip(authors, colors))
        
        # Plot each author
        for author in authors:
            mask = self.aggregated_df['author'] == author
            ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                      c=[color_map[author]], s=150, alpha=0.8, 
                      label=author, edgecolors='white', linewidth=1)
        
        # Add book labels
        for i, row in self.aggregated_df.iterrows():
            ax.annotate(row['book'], 
                       (book_pca[i, 0], book_pca[i, 1]),
                       xytext=(6, 6), textcoords='offset points',
                       fontsize=8, fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('Bible Books by Traditional Author Attribution\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/images/07_author_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_genre_analysis(self):
        """Create visualization showing literary genre groupings."""
        print("📊 Creating literary genre analysis...")
        
        # Add genre information to aggregated data
        self.aggregated_df['genre'] = 'Other'
        for genre, books in self.literary_genres.items():
            mask = self.aggregated_df['book'].isin(books)
            self.aggregated_df.loc[mask, 'genre'] = genre
        
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create color palette for genres
        genres = self.aggregated_df['genre'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(genres)))
        color_map = dict(zip(genres, colors))
        
        # Plot each genre
        for genre in genres:
            mask = self.aggregated_df['genre'] == genre
            ax.scatter(book_pca[mask, 0], book_pca[mask, 1], 
                      c=[color_map[genre]], s=150, alpha=0.8, 
                      label=genre, edgecolors='white', linewidth=1)
        
        # Add book labels
        for i, row in self.aggregated_df.iterrows():
            ax.annotate(row['book'], 
                       (book_pca[i, 0], book_pca[i, 1]),
                       xytext=(6, 6), textcoords='offset points',
                       fontsize=8, fontweight='bold', alpha=0.9,
                       bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, edgecolor='gray'))
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=14)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=14)
        ax.set_title('Bible Books by Literary Genre\n(PCA Projection)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/images/08_genre_analysis.png', dpi=300, bbox_inches='tight')
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
                               'Verses: ' + plot_df['verse_count'].astype(str)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('By Testament', 'By Category', 'By Author', 'By Genre'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Testament plot
        for testament in ['OT', 'NT']:
            mask = plot_df['testament'] == testament
            fig.add_trace(
                go.Scatter(
                    x=plot_df[mask]['PC1'],
                    y=plot_df[mask]['PC2'],
                    mode='markers+text',
                    text=plot_df[mask]['book'],
                    textposition="middle center",
                    textfont=dict(size=8),
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
            title='Interactive Bible Book Analysis (PCA Projection)',
            title_font_size=20,
            height=800,
            showlegend=False
        )
        
        # Save as HTML
        fig.write_html('results/interactive_bible_analysis.html')
        print("✅ Interactive visualization saved as 'results/interactive_bible_analysis.html'")
    
    def create_clean_ot_nt_chart(self):
        """Create a clean, focused OT vs NT chart HTML file with static-style formatting."""
        print("📊 Creating clean OT vs NT chart...")
        
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("⚠️  Plotly not installed. Skipping clean OT vs NT chart.")
            print("   💡 Install with: uv add plotly")
            return
        
        # Apply PCA
        pca = PCA(n_components=2)
        book_pca = pca.fit_transform(self.aggregated_embeddings)
        
        # Prepare dataframe
        plot_df = self.aggregated_df.copy()
        plot_df['PC1'] = book_pca[:, 0]
        plot_df['PC2'] = book_pca[:, 1]
        plot_df['hover_text'] = (
            plot_df['book'] + '<br>' +
            'Testament: ' + plot_df['testament'] + '<br>' +
            'Category: ' + plot_df['category'] + '<br>' +
            'Verses: ' + plot_df['verse_count'].astype(str)
        )
        
        # Colors and marker style
        ot_color = '#2E86AB'
        nt_color = '#A23B72'
        marker_size = 16
        marker_line_width = 2
        
        # Create figure
        fig = go.Figure()
        
        # OT points
        ot_mask = plot_df['testament'] == 'OT'
        fig.add_trace(go.Scatter(
            x=plot_df[ot_mask]['PC1'],
            y=plot_df[ot_mask]['PC2'],
            mode='markers',
            name='Old Testament',
            marker=dict(
                color=ot_color,
                size=marker_size,
                line=dict(color='white', width=marker_line_width),
                opacity=0.85
            ),
            hovertext=plot_df[ot_mask]['hover_text'],
            hoverinfo='text',
            showlegend=True
        ))
        # NT points
        nt_mask = plot_df['testament'] == 'NT'
        fig.add_trace(go.Scatter(
            x=plot_df[nt_mask]['PC1'],
            y=plot_df[nt_mask]['PC2'],
            mode='markers',
            name='New Testament',
            marker=dict(
                color=nt_color,
                size=marker_size,
                line=dict(color='white', width=marker_line_width),
                opacity=0.85
            ),
            hovertext=plot_df[nt_mask]['hover_text'],
            hoverinfo='text',
            showlegend=True
        ))
        # Add book labels as annotations (with white background and gray border)
        for i, row in plot_df.iterrows():
            fig.add_annotation(
                x=row['PC1'],
                y=row['PC2'],
                text=row['book'],
                showarrow=False,
                font=dict(size=11, color='#222', family='Arial Black, Arial, sans-serif'),
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                borderpad=2,
                opacity=0.95,
                xanchor='center',
                yanchor='bottom',
            )
        # Layout
        fig.update_layout(
            title=dict(
                text='<b>(PCA Projection)</b>',
                x=0.5,
                font=dict(size=22, color='#222')
            ),
            xaxis=dict(
                title=dict(
                    text=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                    font=dict(size=16, color='#222', family='Arial')
                ),
                showgrid=True,
                gridcolor='#e5e5e5',
                zeroline=False,
                showline=False,
                tickfont=dict(size=13, color='#222', family='Arial')
            ),
            yaxis=dict(
                title=dict(
                    text=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                    font=dict(size=16, color='#222', family='Arial')
                ),
                showgrid=True,
                gridcolor='#e5e5e5',
                zeroline=False,
                showline=False,
                tickfont=dict(size=13, color='#222', family='Arial')
            ),
            legend=dict(
                x=0.98, y=0.98, xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#bbb', borderwidth=1,
                font=dict(size=14, color='#222', family='Arial')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=900,
            margin=dict(l=80, r=80, t=80, b=80)
        )
        # Save as HTML
        fig.write_html(
            'results/bible_nt_ot_chart.html',
            include_plotlyjs=True,
            full_html=True,
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d', 'resetScale2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'bible_nt_ot_chart',
                    'height': 900,
                    'width': 1200,
                    'scale': 2
                }
            }
        )
        print("✅ Clean OT vs NT chart saved as 'results/bible_nt_ot_chart.html'")
    
    def create_network_analysis(self):
        """Create network analysis showing book relationships."""
        print("📊 Creating network analysis...")
        
        try:
            import networkx as nx
            from matplotlib.patches import Circle
        except ImportError:
            print("⚠️  NetworkX not installed. Skipping network analysis.")
            print("   💡 Already in dependencies - try: uv sync")
            return
        
        # Calculate similarities
        similarities = cosine_similarity(self.aggregated_embeddings)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for i, row in self.aggregated_df.iterrows():
            G.add_node(row['book'], 
                      testament=row['testament'], 
                      category=row['category'],
                      verses=row['verse_count'])
        
        # Add edges for high similarity (top 15% of connections)
        threshold = np.percentile(similarities, 85)
        for i in range(len(similarities)):
            for j in range(i+1, len(similarities)):
                if similarities[i, j] > threshold:
                    G.add_edge(self.aggregated_df.iloc[i]['book'], 
                              self.aggregated_df.iloc[j]['book'],
                              weight=similarities[i, j])
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(20, 16))
        
        # Use spring layout for better organization
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes by testament
        ot_nodes = [node for node in G.nodes() if self.aggregated_df[self.aggregated_df['book'] == node]['testament'].iloc[0] == 'OT']
        nt_nodes = [node for node in G.nodes() if self.aggregated_df[self.aggregated_df['book'] == node]['testament'].iloc[0] == 'NT']
        
        nx.draw_networkx_nodes(G, pos, nodelist=ot_nodes, node_color='#2E86AB', 
                              node_size=800, alpha=0.8, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=nt_nodes, node_color='#A23B72', 
                              node_size=800, alpha=0.8, ax=ax)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Add legend
        ot_patch = Circle((0, 0), 0.1, facecolor='#2E86AB', label='Old Testament')
        nt_patch = Circle((0, 0), 0.1, facecolor='#A23B72', label='New Testament')
        ax.legend(handles=[ot_patch, nt_patch], loc='upper left', bbox_to_anchor=(0, 1))
        
        ax.set_title('Bible Book Relationship Network\n(Connections show high semantic similarity)', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('results/images/09_network_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print network statistics
        print(f"📊 Network Statistics:")
        print(f"   - Nodes: {G.number_of_nodes()}")
        print(f"   - Edges: {G.number_of_edges()}")
        print(f"   - Similarity threshold: {threshold:.3f}")
        print(f"   - Most connected books: {sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]}")
    
    def export_data(self):
        """Export analysis data as CSV/JSON for further use."""
        print("📊 Exporting analysis data...")
        
        # Export book data
        self.aggregated_df.to_csv('results/data/book_analysis.csv', index=False)
        
        # Export chapter data  
        self.chapter_df.to_csv('results/data/chapter_analysis.csv', index=False)
        
        # Export similarity matrices
        book_similarities = cosine_similarity(self.aggregated_embeddings)
        similarity_df = pd.DataFrame(
            book_similarities,
            index=self.aggregated_df['book'],
            columns=self.aggregated_df['book']
        )
        similarity_df.to_csv('results/data/book_similarities.csv')
        
        # Export book categories as JSON
        categories_data = {
            'book_categories': self.book_categories,
            'author_groupings': self.author_groupings,
            'literary_genres': self.literary_genres
        }
        with open('results/data/book_categories.json', 'w') as f:
            json.dump(categories_data, f, indent=2)
        
        print("✅ Data exported to results/data/")

    def analyze_book_similarities(self, target_books=None):
        """Analyze similarities between specific books."""
        if target_books is None:
            target_books = ['Genesis', 'Psalms', 'Isaiah', 'Matthew', 'Romans', 'Revelation']
        
        print(f"📊 Analyzing similarities for: {', '.join(target_books)}")
        
        results = {}
        for target_book in target_books:
            if target_book not in self.aggregated_df['book'].values:
                print(f"⚠️  {target_book} not found in data")
                continue
                
            target_idx = self.aggregated_df[self.aggregated_df['book'] == target_book].index[0]
            target_embedding = self.aggregated_embeddings[target_idx:target_idx+1]
            
            # Calculate similarities to all other books
            similarities = cosine_similarity(target_embedding, self.aggregated_embeddings)[0]
            
            # Get top similar books (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:6]  # Top 5, excluding self
            
            similar_books = []
            for idx in similar_indices:
                book = self.aggregated_df.iloc[idx]['book']
                testament = self.aggregated_df.iloc[idx]['testament']
                category = self.aggregated_df.iloc[idx]['category']
                similarity = similarities[idx]
                similar_books.append((book, testament, category, similarity))
            
            results[target_book] = similar_books
            
            print(f"\n📖 Most similar to {target_book}:")
            for i, (book, testament, category, similarity) in enumerate(similar_books, 1):
                print(f"  {i}. {book} ({testament}, {category}): {similarity:.3f}")
        
        return results

    def write_enhanced_report(self, testament_results, cross_testament_pairs, book_similarities):
        """Write comprehensive analysis report with PNG references."""
        print("📝 Writing enhanced analysis report...")
        
        with open('results/reports/bible_enhanced_analysis.md', 'w') as f:
            f.write("# 📖 Enhanced Bible Embeddings Analysis\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            f.write("**Data Source**: Neon Database (single source of truth)\n\n")
            
            # Dataset Overview
            f.write("## 📊 Dataset Overview\n\n")
            f.write(f"- **Total books analyzed**: {len(self.aggregated_df)}\n")
            f.write(f"- **Old Testament books**: {testament_results['ot_count']}\n")
            f.write(f"- **New Testament books**: {testament_results['nt_count']}\n")
            f.write(f"- **Total chapters**: {len(self.chapter_df)}\n")
            f.write(f"- **Total verses**: {self.aggregated_df['verse_count'].sum():,}\n\n")
            
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
            
            f.write("### 6. Chapter-Level Analysis\n")
            f.write("![Chapter Analysis](../images/06_chapter_analysis.png)\n\n")
            f.write("**Analysis**: How chapters cluster within individual books, showing narrative progression.\n\n")
            
            f.write("### 7. Author Analysis\n")
            f.write("![Author Analysis](../images/07_author_analysis.png)\n\n")
            f.write("**Analysis**: Books grouped by traditional author attribution (Moses, Paul, John, etc.).\n\n")
            
            f.write("### 8. Literary Genre Analysis\n")
            f.write("![Genre Analysis](../images/08_genre_analysis.png)\n\n")
            f.write("**Analysis**: Books categorized by literary genre (Narrative, Poetry, Prophetic, Legal, Epistolary, Apocalyptic).\n\n")
            
            f.write("### 9. Network Analysis\n")
            f.write("![Network Analysis](../images/09_network_analysis.png)\n\n")
            f.write("**Analysis**: Network graph showing high-similarity connections between books.\n\n")
            
            f.write("### 10. Interactive Visualization\n")
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
            f.write("- `../data/chapter_analysis.csv` - Chapter-level data\n")
            f.write("- `../data/book_similarities.csv` - Similarity matrix\n")
            f.write("- `../data/book_categories.json` - Category definitions\n\n")
            
            # Technical Details
            f.write("## 🔧 Technical Details\n\n")
            f.write("### Methodology\n")
            f.write("- **Embedding Model**: Google text-embedding-004 (768 dimensions)\n")
            f.write("- **Dimensionality Reduction**: PCA and t-SNE for visualization\n")
            f.write("- **Similarity Metric**: Cosine similarity\n")
            f.write("- **Clustering**: Natural groupings based on embedding similarities\n")
            f.write("- **Categories**: Traditional biblical literary/theological classifications\n\n")
            
            f.write("### File Structure\n")
            f.write("```\n")
            f.write("results/\n")
            f.write("├── images/          # PNG visualizations\n")
            f.write("├── reports/         # Analysis reports\n")
            f.write("└── data/           # Raw data exports\n")
            f.write("```\n\n")
            
            f.write("---\n")
            f.write("*This enhanced analysis was generated using embeddings-based semantic analysis with multiple visualization techniques.*\n")
        
        print("✅ Enhanced report written to `results/reports/bible_enhanced_analysis.md`")

    def cleanup_old_files(self):
        """Clean up old analysis files to reduce clutter."""
        print("🧹 Cleaning up old analysis files...")
        
        old_files = [
            'bible_analysis_report.md',  # Old report format
            'bible_complete_analysis.md',  # Old location
            'bible_enhanced_analysis.md',  # Old location  
            'bible_books_pca.png',
            'bible_chapters_pca.png', 
            'all_book_chunks_pca.png',
            'genesis_chapters_pca.png',
            'matthew_chapters_pca.png',
            'revelation_chapters_pca.png',
            'three_book_visualizations.png',
            'testament_comparison.png',
            'testament_analysis.png'  # Old single image approach
        ]
        
        old_folders = ['images']  # Old image folder
        
        import os
        import shutil
        
        removed_count = 0
        for file in old_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    removed_count += 1
                    print(f"  ✅ Removed {file}")
            except Exception as e:
                print(f"  ⚠️ Could not remove {file}: {e}")
        
        for folder in old_folders:
            try:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    print(f"  ✅ Removed old folder {folder}/")
            except Exception as e:
                print(f"  ⚠️ Could not remove {folder}/: {e}")
        
        if removed_count > 0:
            print(f"🧹 Cleaned up {removed_count} old files")
        else:
            print("🧹 No old files to clean up")

    def run_complete_analysis(self):
        """Run the complete enhanced analysis pipeline."""
        print("🚀 Starting Enhanced Bible Analysis")
        print("=" * 50)
        
        # Create output directory and clean up old files
        self.create_output_directory()
        self.cleanup_old_files()
        
        # Load data
        self.load_data_from_db()
        
        # Aggregate books
        self.aggregate_books()
        
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
        self.create_chapter_analysis()
        self.create_author_analysis()
        self.create_genre_analysis()
        self.create_interactive_plotly_visualization()
        self.create_clean_ot_nt_chart()
        self.create_network_analysis()
        
        # Export data
        self.export_data()
        
        # Analyze book similarities
        book_similarities = self.analyze_book_similarities()
        
        # Write enhanced report
        self.write_enhanced_report(testament_results, cross_testament_pairs, book_similarities)
        
        print("\n🎉 Enhanced analysis complete!")
        print("📁 Check the `results/` directory:")
        print("   - `images/` - 9 PNG visualizations")
        print("   - `interactive_bible_analysis.html` - Interactive web visualization")
        print("   - `bible_nt_ot_chart.html` - Clean OT vs NT chart")
        print("   - `reports/` - Comprehensive markdown report")
        print("   - `data/` - CSV/JSON exports for further analysis")
        print("\n🎨 New visualizations added:")
        print("   - Author groupings (traditional attributions)")
        print("   - Literary genres (narrative, poetry, prophetic, etc.)")
        print("   - Interactive Plotly charts (web-based)")
        print("   - Network analysis (relationship connections)")
        print("\n📦 Optional features available:")
        print("   - Advanced clustering: `uv add --optional analysis`")
        print("   - Web serving: `uv add --optional web`")
        print("   - Development tools: `uv add --optional dev`")

def main():
    analyzer = BibleAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 