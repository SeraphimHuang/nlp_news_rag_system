import pandas as pd
import numpy as np
import os
import json
import time
import random
import requests
from tqdm import tqdm
from main import NewsRAGSystem, generate_answer_with_ollama

# Configuration
DATA_FILE = 'News_Category_Dataset_v3 2.json'
CHECKPOINT_DIR = './newsrag_checkpoint'
SAMPLE_SIZE = 50 # Set to 20 or 50 as needed
K_RETRIEVAL = 3
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = 'mistral'

class RAGEvaluator:
    def __init__(self):
        print("Initializing RAG System for Evaluation...")
        self.rag = NewsRAGSystem()
        # Load ALL indices
        if not os.path.exists(CHECKPOINT_DIR):
            raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_DIR}. Please run build first.")
        self.rag.load(CHECKPOINT_DIR)
        print("✓ RAG System Loaded")
        
        # No need to load raw dataset separately, we sample from indexed documents
        print(f"✓ Index contains {len(self.rag.documents)} documents")

    def generate_question(self, doc):
        """Use LLM to generate a question from a news document"""
        context = f"Headline: {doc['headline']}\nContent: {doc['short_description']}"
        prompt = f"""Based on the following news snippet, write ONE specific question that can be answered by the content.
        Snippet:
        {context}
        
        Question:"""
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=30
            )
            return response.json()['response'].strip().replace('"', '')
        except Exception as e:
            print(f"⚠️ Error generating question: {e}")
            return None

    def check_hallucination(self, context_text, answer):
        """Use LLM to judge faithfulness"""
        prompt = f"""You are a fact-checker. 
        Context: {context_text}
        
        Claim: {answer}
        
        Does the Claim contain information NOT present in the Context? 
        Reply ONLY with 'YES' (it hallucinates) or 'NO' (it is faithful)."""
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=30
            )
            resp = response.json()['response'].strip().upper()
            return 0 if 'YES' in resp else 1 # 1 means Faithful (Good)
        except:
            return 0.5 # Unknown

    def run_evaluation(self):
        # 1. Sample Documents
        print(f"\nSampling {SAMPLE_SIZE} documents...")
        # We need to know the internal FAISS ID to check retrieval
        # Since we loaded full dataset in main.py logic, self.rag.documents matches retrieval indices IF built correctly
        # BUT: build_index might have sampled or shuffled. 
        # CRITICAL: We must sample from self.rag.documents to ensure IDs match indices!
        
        total_docs = len(self.rag.documents)
        sample_indices = random.sample(range(total_docs), SAMPLE_SIZE)
        
        results = []
        
        for target_id in tqdm(sample_indices, desc="Evaluating"):
            target_doc = self.rag.documents[target_id]
            # Construct passage manually as it might be minimal in 'documents' depending on save logic
            # In our main.py load(), documents dict is restored.
            
            # 2. Generate Question (Ground Truth)
            question = self.generate_question(target_doc)
            if not question: continue
            
            # Define 3 configurations
            configs = [
                {'name': 'Basic', 'strategy': 'simple', 'rerank': False},
                {'name': 'Intermediate', 'strategy': 'simple', 'rerank': True},
                {'name': 'Advanced', 'strategy': 'weighted', 'rerank': True}
            ]
            
            for cfg in configs:
                # 3. Retrieve
                # To calculate MRR properly, we need a larger initial pool before top-k
                # But retrieve returns top-k directly. 
                # Let's use k=10 for checking rank, but metrics usually cut off at k=3
                start_time = time.time()
                retrieved = self.rag.retrieve(
                    question, 
                    strategy=cfg['strategy'], 
                    use_reranker=cfg['rerank'],
                    k=10 # Fetch more to find rank
                )
                latency = (time.time() - start_time) * 1000 # ms
                
                # 4. Calculate Retrieval Metrics
                # Check if target_doc's content matches any retrieved doc
                # Using URL or exact passage match is safer than ID because indices might shift if not careful
                # Let's use URL
                target_url = target_doc.get('link')
                
                rank = float('inf')
                hit_at_3 = 0
                
                for r, res in enumerate(retrieved):
                    if res['document'].get('link') == target_url:
                        rank = r + 1
                        break
                
                if rank <= 3: hit_at_3 = 1
                mrr = 1.0 / rank if rank != float('inf') else 0.0
                
                # 5. Generate Answer (Top-3 context)
                top_3_docs = retrieved[:3]
                gen_out = generate_answer_with_ollama(question, top_3_docs, model=MODEL_NAME)
                answer_text = gen_out['answer']
                
                # 6. Judge Faithfulness
                # Context is the joined text of top-3
                context_text = "\n".join([d['passage'] for d in top_3_docs])
                faithfulness = self.check_hallucination(context_text, answer_text)
                
                results.append({
                    'Config': cfg['name'],
                    'Target_ID': target_id,
                    'Question': question,
                    'Target_URL': target_url,
                    'Rank': rank if rank != float('inf') else -1,
                    'Recall@3': hit_at_3,
                    'MRR': mrr,
                    'Latency_ms': latency,
                    'Faithfulness': faithfulness,
                    'Answer': answer_text,
                    'Context_Found': rank <= 10
                })
                
        # Save Results
        df_res = pd.DataFrame(results)
        df_res.to_csv('evaluation_results.csv', index=False)
        
        print("\n" + "="*50)
        print("EVALUATION REPORT")
        print("="*50)
        print(df_res.groupby('Config')[['Recall@3', 'MRR', 'Latency_ms', 'Faithfulness']].mean())
        print("\nDetailed results saved to evaluation_results.csv")

if __name__ == "__main__":
    # Ensure random seed for reproducibility
    random.seed(42)
    evaluator = RAGEvaluator()
    evaluator.run_evaluation()

