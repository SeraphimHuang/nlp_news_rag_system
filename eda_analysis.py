import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Configuration
DATA_FILE = 'News_Category_Dataset_v3 2.json'
OUTPUT_DIR = 'eda_results'

def run_eda():
    print(f"Starting EDA on {DATA_FILE}...")
    
    # 1. Check file
    if not os.path.exists(DATA_FILE):
        print(f"❌ Error: File {DATA_FILE} not found!")
        return

    # 2. Load Data
    try:
        df = pd.read_json(DATA_FILE, lines=True)
        print(f"✓ Loaded {len(df)} rows")
    except Exception as e:
        print(f"❌ Error loading json: {e}")
        return

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)

    # =================================================================
    # ANALYSIS 1: Category Distribution
    # =================================================================
    print("Analyzing Category Distribution...")
    plt.figure(figsize=(14, 8))
    
    # Count plot
    category_counts = df['category'].value_counts()
    sns.barplot(x=category_counts.values, y=category_counts.index, palette="viridis")
    
    plt.title('Distribution of News Categories', fontsize=16)
    plt.xlabel('Number of Articles')
    plt.ylabel('Category')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eda_1_categories.png')
    print(f"✓ Saved {OUTPUT_DIR}/eda_1_categories.png")
    plt.close()

    # =================================================================
    # ANALYSIS 2: Text Length Analysis
    # =================================================================
    print("Analyzing Text Lengths...")
    
    # Calculate lengths (word count)
    df['headline_len'] = df['headline'].apply(lambda x: len(str(x).split()))
    df['desc_len'] = df['short_description'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    sns.histplot(df['headline_len'], color='skyblue', label='Headline', bins=30, alpha=0.6)
    sns.histplot(df['desc_len'], color='orange', label='Description', bins=30, alpha=0.6)
    
    plt.title('Distribution of Word Counts (Headline vs Description)', fontsize=16)
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eda_2_text_lengths.png')
    print(f"✓ Saved {OUTPUT_DIR}/eda_2_text_lengths.png")
    plt.close()
    
    # Print stats
    print(f"  Headline Avg Length: {df['headline_len'].mean():.1f} words")
    print(f"  Headline Min/Max Length: {df['headline_len'].min()}/{df['headline_len'].max()} words")
    print(f"  Description Avg Length: {df['desc_len'].mean():.1f} words")
    print(f"  Description Min/Max Length: {df['desc_len'].min()}/{df['desc_len'].max()} words")

    # =================================================================
    # ANALYSIS 3: Temporal Analysis
    # =================================================================
    print("Analyzing Temporal Trends...")
    
    # Convert to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Resample by month
    monthly_counts = df.set_index('date').resample('M').size()
    
    plt.figure(figsize=(14, 6))
    sns.lineplot(x=monthly_counts.index, y=monthly_counts.values, linewidth=2.5, color='purple')
    
    plt.title('Number of News Articles Over Time (Monthly)', fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Article Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/eda_3_temporal.png')
    print(f"✓ Saved {OUTPUT_DIR}/eda_3_temporal.png")
    plt.close()

    print("\nEDA Complete! Check the 'eda_results' folder.")

if __name__ == "__main__":
    run_eda()

