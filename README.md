# NewsRAG: Local Retrieval-Augmented Generation System

A privacy-focused, local-first RAG (Retrieval-Augmented Generation) system designed to answer questions based on news datasets. This project leverages **Ollama** for local LLM inference and **FAISS** for efficient vector retrieval, ensuring that no data leaves your local environment.

## 🚀 Features

*   **Local LLM Inference**: Uses [Ollama](https://ollama.ai/) to run state-of-the-art open models (like Mistral, Llama 2) locally.
*   **Vector Search**: Implements FAISS for high-performance similarity search over news articles.
*   **Temporal Context Awareness**: Decoupled indexing and generation logic allows the LLM to access timestamped information while maintaining pure semantic search.
*   **Interactive UI**: Provides a clean Streamlit interface for easy interaction.
*   **Dockerized**: Fully containerized for one-click deployment and reproducibility.

## 🛠️ System Architecture

The system follows a standard RAG pipeline:
1.  **Data Processing**: Loads news dataset, cleans text, and prepares passages.
2.  **Indexing**: Encodes text using `sentence-transformers/all-MiniLM-L6-v2` and builds a Flat-IP FAISS index.
3.  **Retrieval**: Searches the vector database for the top-k most relevant articles based on semantic similarity.
4.  **Generation**: Constructs a prompt containing the retrieved articles (enriched with temporal metadata) and queries the local Ollama instance for a concise answer.

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
Data source: https://www.kaggle.com/datasets/rmisra/news-category-dataset
Misra, Rishabh. "News Category Dataset." arXiv preprint arXiv:2209.11429 (2022)
*(Note: If the file is not present, the system will look for it at runtime. Ensure you have the dataset available.)*

### 3. Build the Index (Required)
Before running the application, you must process the dataset and build the vector index. This prevents the UI from hanging during startup.

```bash
# Using Docker (Recommended)
docker-compose run --rm rag-app python main.py --data "News_Category_Dataset_v3 2.json" --sample-size 50000 --save-index ./newsrag_checkpoint

# Or running locally
python main.py --data "News_Category_Dataset_v3 2.json" --sample-size 50000 --save-index ./newsrag_checkpoint
```

### 4. Start Services
```bash
docker-compose up
```

This command will:
1.  Start the **Ollama** container (serving the LLM).
2.  Start the **NewsRAG** app container (loading the index you just built).

### 5. Initialize Model
In the UI sidebar or terminal, ensure you have pulled the required model:
```bash
# Inside the ollama container or via local CLI if installed
docker exec -it ollama-service ollama pull mistral
```
*(The UI will guide you if the model is missing)*

### 6. Access the UI
Open your browser and navigate to:
`http://localhost:8501`

## 🔧 Design Decisions

### 1. Local-First Approach (Ollama)
We chose Ollama over cloud APIs (like OpenAI) to ensure complete data privacy and zero cost for inference. This allows the system to run entirely offline once the models are downloaded.

### 2. Decoupling Indexing from Metadata
To improve temporal reasoning, we separated the embedding text from the generation context:
*   **Index**: Only semantic content (`headline` + `short_description`) is embedded. This prevents dates from skewing semantic similarity.
*   **Prompt**: The `date` field is dynamically injected into the LLM prompt during generation (e.g., `[2022-09-23] ...`). This allows the model to answer questions like "What happened *after* September?" accurately.

### 3. Containerization Strategy
We use a multi-container setup:
*   `ollama`: Dedicated to model serving, allowing it to leverage GPU resources independently.
*   `rag-app`: Contains the application logic, ensuring the frontend/backend code is isolated from the inference engine.

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

The system's effectiveness relies on the `all-MiniLM-L6-v2` model for high-quality dense retrieval and the `mistral` (or similar) model for reasoning. The "brute-force" Flat-IP index in FAISS ensures 100% recall for retrieval, suitable for datasets up to millions of documents.

