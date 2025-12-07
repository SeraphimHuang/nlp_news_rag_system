# NewsRAG: Local Retrieval-Augmented Generation System

A privacy-focused, local-first RAG (Retrieval-Augmented Generation) system designed to answer questions based on news datasets. This project leverages **Ollama** for local LLM inference, **FAISS** for efficient vector retrieval, and **Cross-Encoders** for high-precision re-ranking.

## 🚀 Features

*   **Local LLM Inference**: Uses [Ollama](https://ollama.ai/) to run state-of-the-art open models (like Mistral, Llama 2) locally.
*   **Two-Stage Retrieval**: Combines fast Bi-Encoder retrieval with high-precision Cross-Encoder re-ranking for superior accuracy.
*   **Temporal Context Awareness**: Decoupled indexing and generation logic allows the LLM to access timestamped information while maintaining pure semantic search.
*   **Interactive UI**: Provides a clean Streamlit interface for easy interaction.
*   **Dockerized**: Fully containerized for one-click deployment and reproducibility.

## 🛠️ System Architecture

The system follows an advanced RAG pipeline:
1.  **Data Processing**: Loads news dataset, cleans text, and prepares passages.
2.  **Indexing**: Encodes text using `sentence-transformers/all-MiniLM-L6-v2` and builds a Flat-IP FAISS index.
3.  **Retrieval & Re-ranking**: 
    *   **Step 1 (Recall)**: Retrieves the top-50 most relevant candidates using fast vector search (Bi-Encoder).
    *   **Step 2 (Precision)**: Re-ranks these 50 candidates using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to select the top-k highest quality passages.
4.  **Generation**: Constructs a prompt containing the refined articles (enriched with temporal metadata) and queries the local Ollama instance.

## 📋 Prerequisites

*   [Docker](https://www.docker.com/) & Docker Compose
*   (Optional) NVIDIA GPU support for faster inference (recommended)

## 🏃‍♂️ Quick Start (Reproducibility)

The easiest way to run the system is via Docker Compose. This will automatically set up the Ollama service, build the application, and start the Streamlit interface.

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

### 3. Build the Index (Required)
Before running the application, you must process the dataset and build the vector index. **Crucial:** We mount the current directory to persist the index file to your local machine so it survives container restarts.

```bash
# 1. Build the Docker image
docker-compose build

# 2. Run the indexing script
# The -v "$(pwd):/app" flag ensures the generated index is saved to your host machine
docker-compose run --rm -v "$(pwd):/app" rag-app python main.py --data "News_Category_Dataset_v3 2.json" --sample-size 50000 --save-index ./newsrag_checkpoint
```

*Note: You can adjust `--sample-size` as needed (e.g., 100000). The process may take a few minutes.*

### 4. Start Services
Once the index is built (you should see a `newsrag_checkpoint` folder appear), start the full system:

```bash
docker-compose up
```

This command will:
1.  Start the **Ollama** container (serving the LLM).
2.  Start the **NewsRAG** app container (loading the persisted index).

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

## 🔧 Design Decisions

### 1. Retrieve-then-Rerank Architecture
To balance speed and accuracy, we implement a two-stage retrieval process. The Bi-Encoder (FAISS) acts as a fast filter to reduce millions of documents to a small candidate set (50). The Cross-Encoder then acts as a rigorous judge, scoring each candidate pair-wise against the query. This significantly boosts the relevance of the context provided to the LLM.

### 2. Local-First Approach (Ollama)
We chose Ollama over cloud APIs (like OpenAI) to ensure complete data privacy and zero cost for inference. This allows the system to run entirely offline once the models are downloaded.

### 3. Decoupling Indexing from Metadata
To improve temporal reasoning, we separated the embedding text from the generation context:
*   **Index**: Only semantic content (`headline` + `short_description`) is embedded.
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

The system uses `all-MiniLM-L6-v2` for dense retrieval and `cross-encoder/ms-marco-MiniLM-L-6-v2` for re-ranking. This combination allows for "brute-force" 100% recall in the first stage, followed by high-precision filtering, ensuring the LLM receives the most pertinent information.
