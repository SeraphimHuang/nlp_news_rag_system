#!/usr/bin/env python3
"""
NewsRAG: Local Retrieval-Augmented Generation System for News QA
Supports local Ollama LLM backend
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import requests
from typing import List, Tuple, Dict, Optional
import time
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss
except ImportError:
    print("Installing required packages...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
                          'sentence-transformers', 'faiss-cpu', 'pandas', 'numpy'])
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import faiss

# ============================================================================
# PART 1: DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess_data(data_path: str, sample_size: int = 5000) -> pd.DataFrame:
    """Load and preprocess news dataset from JSON or CSV"""
    
    print(f"\n{'='*70}")
    print("STEP 1: DATA PREPROCESSING")
    print('='*70)
    
    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        return None
    
    print(f"Loading data from {data_path}...")
    file_size = os.path.getsize(data_path) / (1024*1024)
    print(f"File size: {file_size:.1f} MB")
    
    try:
        if data_path.endswith('.json'):
            df = pd.read_json(data_path, lines=True)
        else:
            df = pd.read_csv(data_path)
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return None
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Clean the data
    df = df.dropna(subset=['headline', 'short_description'])
    df = df[df['headline'].str.len() > 0]
    df = df[df['short_description'].str.len() > 0]
    
    # Create passage
    df['passage'] = df['headline'] + ' ' + df['short_description']
    df = df.drop_duplicates(subset=['passage'])
    df['passage'] = df['passage'].str.lower().str.strip()
    
    # Sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"✓ Preprocessed: {len(df)} documents")
    print(f"Sample passage:\n  {df['passage'].iloc[0][:100]}...\n")
    
    return df.reset_index(drop=True)

# ============================================================================
# PART 2: EMBEDDING GENERATION & FAISS INDEXING
# ============================================================================

class NewsRAGSystem:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', 
                 reranker_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 device: str = 'cpu'):
        """
        Initialize RAG system
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        
        print(f"Loading reranker model: {reranker_name}")
        self.reranker = CrossEncoder(reranker_name, device=device)
        
        # Initialize ALL index holders
        self.faiss_index = None          # For 'simple' strategy
        self.faiss_index_headline = None # For 'weighted' strategy
        self.faiss_index_content = None  # For 'weighted' strategy
        
        self.documents = []
        self.passages = []
        self.urls = []

    def build_index(self, df: pd.DataFrame, batch_size: int = 32):
        """Build ALL FAISS indices (Simple + Weighted) at once"""
        print(f"\n{'='*70}")
        print("STEP 2: BUILD RETRIEVAL INDICES (All Strategies)")
        print('='*70)
        
        self.passages = df['passage'].tolist()
        self.urls = df['link'].tolist()
        self.documents = df.to_dict('records')
        
        # 1. Simple Strategy Index
        print("\n[1/3] Building Index for Simple Strategy (Passages)...")
        self.faiss_index = self._create_faiss_index(self.passages, batch_size)
        
        # 2. Weighted Strategy Indices
        print("\n[2/3] Building Index for Weighted Strategy (Headlines)...")
        headlines = df['headline'].tolist()
        self.faiss_index_headline = self._create_faiss_index(headlines, batch_size)
        
        print("\n[3/3] Building Index for Weighted Strategy (Content)...")
        contents = df['short_description'].tolist()
        self.faiss_index_content = self._create_faiss_index(contents, batch_size)
            
        print("\n✓ All indexing complete\n")

    def _create_faiss_index(self, texts: List[str], batch_size: int):
        embeddings = []
        total = len(texts)
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True)
            embeddings.append(batch_embeddings)
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"    Processed {min(i + batch_size, total)}/{total}")
        
        embeddings = np.vstack(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        return index
    
    def retrieve(self, query: str, strategy: str = 'simple', use_reranker: bool = True, k: int = 5, initial_k: int = 50) -> List[Dict]:
        """Retrieve top-k relevant passages with optional Re-ranking"""
        # 1. Encode Query
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        candidates_map = {} # {idx: score}

        if strategy == 'weighted':
            search_k = min(initial_k * 2, len(self.documents)) if self.documents else initial_k
            
            # Search Headlines (Weight: 0.7)
            if self.faiss_index_headline:
                D_h, I_h = self.faiss_index_headline.search(query_embedding, search_k)
                for idx, score in zip(I_h[0], D_h[0]):
                    idx = int(idx)
                    if idx < 0: continue
                    candidates_map[idx] = candidates_map.get(idx, 0.0) + (float(score) * 0.7)

            # Search Content (Weight: 0.3)
            if self.faiss_index_content:
                D_c, I_c = self.faiss_index_content.search(query_embedding, search_k)
                for idx, score in zip(I_c[0], D_c[0]):
                    idx = int(idx)
                    if idx < 0: continue
                    candidates_map[idx] = candidates_map.get(idx, 0.0) + (float(score) * 0.3)
                
        else: # 'simple'
            search_k = min(initial_k, len(self.documents)) if self.documents else initial_k
            if self.faiss_index:
                D, I = self.faiss_index.search(query_embedding, search_k)
                for idx, score in zip(I[0], D[0]):
                    idx = int(idx)
                    if idx < 0: continue
                    candidates_map[idx] = float(score)

        # 2. Prepare candidates
        # Select top-initial_k based on fused score
        sorted_indices = sorted(candidates_map.items(), key=lambda item: item[1], reverse=True)[:initial_k]
        
        candidates = []
        candidate_pairs = [] # For Cross-Encoder
        
        for idx, fused_score in sorted_indices:
            passage = self.passages[idx]
            
            if use_reranker:
                # Only prepare pairs if we are going to use them
                candidate_pairs.append([query, passage])
            
            # Safely get document
            if idx < len(self.documents):
                document = self.documents[idx]
            else:
                document = {'passage': passage, 'url': self.urls[idx]}
                
            candidates.append({
                'passage': passage,
                'url': self.urls[idx],
                'document': document,
                'score': fused_score # Default to Bi-Encoder score
            })

        if not candidates:
            return []

        # 3. Cross-Encoder Re-ranking (Optional)
        if use_reranker:
            # Predict scores for (Query, Document) pairs
            cross_scores = self.reranker.predict(candidate_pairs)
            
            # Overwrite scores with Cross-Encoder scores
            for i, score in enumerate(cross_scores):
                candidates[i]['score'] = float(score)
            
        # 4. Sort and Top-K
        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)[:k]
        
        # Add rank
        for i, res in enumerate(candidates):
            res['rank'] = i + 1
            
        return candidates
    
    def save(self, save_dir: str = './newsrag_checkpoint'):
        """Save ALL indices"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save indices
        if self.faiss_index:
            faiss.write_index(self.faiss_index, os.path.join(save_dir, 'faiss.index'))
        if self.faiss_index_headline:
            faiss.write_index(self.faiss_index_headline, os.path.join(save_dir, 'faiss_headline.index'))
        if self.faiss_index_content:
            faiss.write_index(self.faiss_index_content, os.path.join(save_dir, 'faiss_content.index'))
            
        # Save metadata
        metadata = {
            'passages': self.passages, 
            'urls': self.urls,
            'documents': self.documents if hasattr(self, 'documents') else []
        }
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✓ System saved to {save_dir}")

    def load(self, save_dir: str = './newsrag_checkpoint'):
        """Load ALL indices"""
        # Load metadata
        with open(os.path.join(save_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.passages = metadata['passages']
        self.urls = metadata['urls']
        self.documents = metadata.get('documents', [])
        
        # Load indices if they exist
        if os.path.exists(os.path.join(save_dir, 'faiss.index')):
            self.faiss_index = faiss.read_index(os.path.join(save_dir, 'faiss.index'))
            
        if os.path.exists(os.path.join(save_dir, 'faiss_headline.index')):
            self.faiss_index_headline = faiss.read_index(os.path.join(save_dir, 'faiss_headline.index'))
            
        if os.path.exists(os.path.join(save_dir, 'faiss_content.index')):
            self.faiss_index_content = faiss.read_index(os.path.join(save_dir, 'faiss_content.index'))

        # Fallback doc creation
        if not self.documents:
            self.documents = [{'passage': p, 'url': u} for p, u in zip(self.passages, self.urls)]
            
        print(f"✓ System loaded from {save_dir}")

# ============================================================================
# PART 3: LLM BACKEND - OLLAMA
# ============================================================================

def check_ollama_connection(base_url: str = None) -> bool:
    """Check if Ollama is running"""
    if base_url is None:
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def list_ollama_models(base_url: str = None) -> List[str]:
    """List available Ollama models"""
    if base_url is None:
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
    except:
        pass
    return []

def generate_answer_with_ollama(query: str, retrieved_docs: List[Dict], 
                                model: str = 'mistral',
                                base_url: str = None) -> Dict:
    """Generate answer using Ollama"""
    if base_url is None:
        base_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    
    context_list = []
    for i, doc in enumerate(retrieved_docs):
        date_str = doc.get('document', {}).get('date', '')
        date_prefix = f"[{date_str}] " if date_str else ""
        context_list.append(f"[{i+1}] {date_prefix}{doc['passage']}")
    
    context = "\n".join(context_list)
    
    system_prompt = """You are a helpful news analyst. Answer questions based on the provided news context.
- Be concise (2-3 sentences)
- Always cite sources using [1], [2], etc.
- If a source does not provide relevant information, don't mention it in the answer.
- Only use information from the context and try to include date information if available.
- If no relevant info, say so"""
    
    prompt = f"""Context from news articles:
{context}

Question: {query}

Answer:"""
    
    try:
        print(f"Generating answer with Ollama ({model})...")
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            answer = response.json()['response'].strip()
        else:
            answer = generate_template_answer(query, retrieved_docs)
            
    except Exception as e:
        print(f"⚠️  Ollama error: {e}")
        answer = generate_template_answer(query, retrieved_docs)
    
    return {
        'query': query,
        'answer': answer,
        'sources': [{'rank': i+1, 'url': doc['url'], 
                     'date': doc.get('document', {}).get('date', 'N/A'),
                     'passage': doc['passage'][:150]} 
                   for i, doc in enumerate(retrieved_docs)]
    }

def generate_template_answer(query: str, retrieved_docs: List[Dict]) -> str:
    """Fallback template-based answer"""
    if not retrieved_docs:
        return "No relevant information found."
    
    passages = "\n".join([f"[{i+1}] {doc['passage'][:120]}..." 
                         for i, doc in enumerate(retrieved_docs)])
    answer = f"Based on retrieved news:\n{passages}"
    return answer

# ============================================================================
# PART 4: INTERACTIVE PIPELINE
# ============================================================================

def interactive_rag(rag_system: NewsRAGSystem, 
                    model: str = 'mistral',
                    base_url: str = "http://localhost:11434",
                    k: int = 3,
                    default_strategy: str = 'simple'):
    """Interactive RAG query loop"""
    
    print(f"\n{'='*70}")
    print(f"INTERACTIVE NEWS RAG SYSTEM (Strategy: {default_strategy})")
    print('='*70)
    
    # ... (Ollama checks) ...
    
    print(f"Using model: {model}\n")
    
    while True:
        print("-" * 70)
        query = input("Enter your question (or 'quit' to exit):\n> ").strip()
        
        # ... (Exit checks) ...
        
        print("\n" + "="*70)
        
        # Retrieve
        start_time = time.time()
        # Pass strategy here
        retrieved = rag_system.retrieve(query, strategy=default_strategy, k=k)
        retrieval_time = time.time() - start_time
        
        # ... (Print results) ...
        
        print(f"Retrieved {len(retrieved)} passages ({retrieval_time:.3f}s):")
        for doc in retrieved:
            print(f"\n[{doc['rank']}] Score: {doc['score']:.4f}")
            print(f"    {doc['passage'][:100]}...")
            print(f"    URL: {doc['url']}")
        
        # Generate
        print("\nGenerating answer...")
        result = generate_answer_with_ollama(query, retrieved, model, base_url)
        
        print(f"\n{'='*70}")
        print("ANSWER:")
        print(result['answer'])
        print(f"{'='*70}")
        
        print("\nSources:")
        for src in result['sources']:
            print(f"  [{src['rank']}] {src['url']}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NewsRAG System')
    parser.add_argument('--data', type=str, required=False,
                       help='Path to News Category Dataset (JSON or CSV)')
    parser.add_argument('--sample-size', type=int, default=5000,
                       help='Number of documents to use')
    parser.add_argument('--model', type=str, default='mistral',
                       help='Ollama model to use')
    parser.add_argument('--ollama-url', type=str, default=os.getenv("OLLAMA_URL", "http://localhost:11434"),
                       help='Ollama server URL')
    parser.add_argument('--save-index', type=str, default=None,
                       help='Save FAISS index to directory')
    parser.add_argument('--load-index', type=str, default=None,
                       help='Load FAISS index from directory')
    parser.add_argument('--k', type=int, default=3,
                       help='Number of documents to retrieve')
    parser.add_argument('--strategy', type=str, default='simple', choices=['simple', 'weighted'],
                       help='Default retrieval strategy for interactive mode')
    
    args = parser.parse_args()
    
    print("\n🚀 NewsRAG System - Local Ollama Backend\n")
    
    # Load or build system
    if args.load_index and os.path.exists(args.load_index):
        print(f"Loading existing index from {args.load_index}...")
        rag = NewsRAGSystem()
        rag.load(args.load_index)
    else:
        # Check if data is provided
        if not args.data:
            print("❌ Error: --data is required when not using --load-index")
            parser.print_help()
            exit(1)
        
        # Load data
        df = load_and_preprocess_data(args.data, args.sample_size)
        if df is None:
            exit(1)
        
        # Build system
        rag = NewsRAGSystem()
        rag.build_index(df)
        
        # Save if requested
        if args.save_index:
            rag.save(args.save_index)
    
    # Interactive loop
    interactive_rag(rag, model=args.model, base_url=args.ollama_url, k=args.k, default_strategy=args.strategy)