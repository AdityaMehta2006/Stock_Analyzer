# AI Equity Analyst

An advanced stock analysis dashboard that leverages Machine Learning (Random Forest, XGBoost) and Deep Learning (LSTM, GRU) to predict stock (equity) movements.

## Features

*   **Interactive Dashboard**: Built with Streamlit for real-time visualization.
*   **Multi-Model Architecture**:
    *   **Standard ML**: Random Forest, XGBoost, Logistic Regression.
    *   **Deep Learning**: LSTM and GRU neural networks for sequential pattern recognition.
    *   **Ensemble**: Stacked and Voting ensembles for robust predictions.
*   **AI Insights (RAG)**:
    *   **Hybrid Intelligence**: Combines quantitative data with qualitative news analysis.
    *   **Local Privacy**: Runs entirely offline using **Ollama (Phi-3)** and **ChromaDB**.
    *   **Multi-Source**: Aggregates news from Yahoo Finance and DuckDuckGo.
*   **Cross-Platform**:
    *   **NVIDIA GPUs**: Accelerated via CUDA.
    *   **Apple Silicon**: Accelerated via MPS (Metal Performance Shaders).
    *   **CPU**: Automatic fallback for widespread compatibility.

## Installation

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install streamlit pandas numpy yfinance scikit-learn xgboost torch plotly
    ```

## Usage

### Run the Dashboard
To start the web application:
```bash
streamlit run src/dashboard.py
```

### Utility Scripts
Check stock availability or test GPU status:
```bash
python scripts/debug_yf.py
python scripts/gpu_test.py
```

## Project Structure

*   `src/`: Main application code (`dashboard.py`, `stock_model.py`).
*   `scripts/`: Utilities for testing connection and hardware.
*   `data/`: Directory for storing downloaded market data CSVs.
*   `docs/`: Detailed mathematical documentation and setup guides.

## Documentation

*   [Model Mathematics](docs/model_math.md): Detailed explanation of the equations and algorithms.
*   [Setup Steps](docs/setup_steps.md): Expanded installation and portability guide.
