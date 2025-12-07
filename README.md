# NewsRAG: Local Retrieval-Augmented Generation System

A privacy-focused, local-first RAG (Retrieval-Augmented Generation) system designed to answer questions based on news datasets. This project leverages **Ollama** for local LLM inference, **FAISS** for efficient vector retrieval, and **Cross-Encoders** for high-precision re-ranking.

## 🚀 Features

*   **Local LLM Inference**: Uses [Ollama](https://ollama.ai/) to run state-of-the-art open models (like Mistral, Llama 2) locally.
*   **Dual-Strategy Retrieval**: Supports both **Simple** (Single Index) and **Weighted** (Dual Index) retrieval strategies, switchable via UI.
*   **Two-Stage Pipeline**: Combines fast Bi-Encoder retrieval with high-precision Cross-Encoder re-ranking.
*   **Temporal Context Awareness**: Decoupled indexing and generation logic allows the LLM to access timestamped information while maintaining pure semantic search.
*   **Interactive UI**: Clean Streamlit interface for easy interaction and strategy comparison.
*   **Dockerized**: Fully containerized for one-click deployment and reproducibility.

## 🛠️ System Architecture

The system follows an advanced RAG pipeline with configurable strategies:

1.  **Data Processing**: Loads news dataset, cleans text, and prepares passages.
2.  **Indexing (Configurable)**:
    *   **Simple Mode**: Builds a single FAISS index on `headline + description`.
    *   **Weighted Mode**: Builds **two separate indices** (one for headlines, one for content) to allow fine-grained weighted retrieval (e.g., 30% Headline + 70% Content).
3.  **Retrieval & Re-ranking**: 
    *   **Step 1 (Recall)**: Retrieves top-50 candidates using Bi-Encoder. In Weighted mode, it fuses scores from both indices.
    *   **Step 2 (Precision)**: Re-ranks candidates using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to select the top-k highest quality passages.
4.  **Generation**: Constructs a prompt containing the refined articles (enriched with temporal metadata) and queries the local Ollama instance.

## 📋 Prerequisites

*   [Docker](https://www.docker.com/) & Docker Compose
*   (Optional) NVIDIA GPU support for faster inference (recommended)

## 🏃‍♂️ Quick Start (Reproducibility)

The easiest way to run the system is via Docker Compose.

### 1. Clone the Repository
```bash
git clone https://github.com/SeraphimHuang/nlp_news_rag_system
cd nlp_news_rag_system
```

### 2. Download Data
Place your `News_Category_Dataset_v3 2.json` file in the project root directory.
*   Data source: https://www.kaggle.com/datasets/rmisra/news-category-dataset
*   Reference: Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022)

*(Note: If the file is not present, the system will look for it at runtime. Ensure you have the dataset available.)*

### 3. Build the Indices (Required)
To enable the strategy switcher in the UI, you should build indices for both strategies.

```bash
# 1. Build the Docker image
docker-compose build

# 2. Build 'Simple' Index (Standard RAG)
docker-compose run --rm -v "$(pwd):/app" rag-app python main.py --strategy simple --save-index ./newsrag_checkpoint --sample-size 50000

# 3. Build 'Weighted' Index (Dual Index Strategy)
docker-compose run --rm -v "$(pwd):/app" rag-app python main.py --strategy weighted --save-index ./newsrag_checkpoint_weighted --sample-size 50000
```
*Note: The `-v "$(pwd):/app"` flag persists the generated indices to your local machine.*

### 4. Start Services
```bash
docker-compose up
```

This command will:
1.  Start the **Ollama** container (serving the LLM).
2.  Start the **NewsRAG** app container (loading the persisted indices).

### 5. Initialize Model
In the UI sidebar or terminal, ensure you have pulled the required model:
```bash
# Inside the ollama container or via local CLI if installed
docker exec -it ollama-service ollama pull mistral
```
*(The UI will also guide you if the model is missing)*

### 6. Access the UI
Open your browser and navigate to:
`http://localhost:8501`

You can now toggle between **Simple** and **Weighted** strategies in the sidebar to compare results.

## 🔧 Design Decisions

### 1. Multi-Index Weighted Retrieval
We observed that standard RAG often misses relevant details because headlines can be clickbait or vague. Our **Weighted Strategy** builds separate indices for Headlines and Content. This allows us to apply a fusion formula (e.g., `Score = 0.3 * Head + 0.7 * Content`), ensuring that a document is retrieved if it has a highly relevant detailed description, even if the headline is obscure.

### 2. Retrieve-then-Rerank Architecture
To balance speed and accuracy, we implement a two-stage retrieval process. The Bi-Encoder (FAISS) acts as a fast filter to reduce millions of documents to a small candidate set (50). The Cross-Encoder then acts as a rigorous judge, scoring each candidate pair-wise against the query.

### 3. Decoupling Indexing from Metadata
To improve temporal reasoning, we separated the embedding text from the generation context:
*   **Index**: Only semantic content is embedded.
*   **Prompt**: The `date` field is dynamically injected into the LLM prompt during generation (e.g., `[2022-09-23] ...`).

## 📂 Project Structure

```
.
├── main.py                # Core RAG logic (Indexing, Retrieval, Generation)
├── streamlit_app.py       # Interactive Frontend
├── Dockerfile             # App container definition
├── docker-compose.yml     # Orchestration
├── requirements.txt       # Python dependencies
└── News_Category_Dataset_v3 2.json # Dataset (input)
```

## 🧪 Evaluation & Methodology

The system uses `all-MiniLM-L6-v2` for dense retrieval and `cross-encoder/ms-marco-MiniLM-L-6-v2` for re-ranking. The "brute-force" Flat-IP index in FAISS ensures 100% recall for retrieval. The dual-index strategy specifically addresses the "information density imbalance" between news headlines and body text.
