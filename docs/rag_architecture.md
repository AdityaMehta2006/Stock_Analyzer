# RAG Architecture & Data Pipelines

## What is RAG?
**Retrieval-Augmented Generation (RAG)** is a technique that enhances the capabilities of a Large Language Model (LLM) by providing it with fresh, external data before it generates an answer.

Instead of relying solely on its training data (which cuts off at a past date), the system:
1.  **Retrieves** relevant information from a knowledge base (in our case, recent stock news).
2.  **Augments** the user's prompt with this retrieved context.
3.  **Generates** a grounded, accurate response using the LLM (Phi-3).

---

## Data Pipelines

### 1. Ingestion Pipeline (News -> VectorDB)
This pipeline triggers when you click **"Fetch & Vectorize News"**.

1.  **Extraction**:
    *   **Source A (Yahoo Finance)**: Fetches official news feeds via `yfinance`.
    *   **Source B (DuckDuckGo)**: Performs a web search for "[Ticker] stock news" to catch wider coverage.
2.  **Transformation**:
    *   Data is standardized into a common format: `{'title', 'summary', 'published', 'source', 'link'}`.
    *   Text is cleaned and formatted into "chunks" (e.g., `[Source] Title: ... Summary: ...`).
3.  **Embedding**:
    *   Each chunk is passed through an **Embedding Model** (`all-MiniLM-L6-v2`).
    *   This converts text into a 384-dimensional vector (a list of numbers representing meaning).
4.  **Loading**:
    *   Vectors are stored in **ChromaDB** (a local vector database) alongside their metadata.
    *   *Optimization*: Old data for the specific ticker is wiped before insertion to prevent context bloat.

### 2. Inference Pipeline (User Query -> Answer)
This pipeline triggers when you ask a question in the "AI Insights" tab.

1.  **Query Embedding**:
    *   The user's question (e.g., "Why is it falling?") is converted into a vector using the same model.
2.  **Semantic Search**:
    *   ChromaDB finds the top `K` news chunks that are mathematically closest (most relevant) to the question's vector.
3.  **Prompt Construction**:
    *   A prompt is built dynamically:
        ```text
        "You are a financial analyst. Use these news snippets: [News 1] [News 2] ... 
         to answer this question: Why is it falling?"
        ```
4.  **Generation**:
    *   The prompt is sent to the local **Ollama** instance running **Phi-3**.
    *   The model generates a concise answer citing the provided sources.

---

## Tech Stack
*   **LLM**: Phi-3 (via Ollama) - 4-bit Quantized.
*   **Vector Database**: ChromaDB (Local).
*   **Embeddings**: sentence-transformers/all-MiniLM-L6-v2.
*   **Orchestration**: Python `requests` & Custom Logic.
