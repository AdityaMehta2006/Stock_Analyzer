# Project Setup and Usage

## Prerequisites

Ensure you have Python installed (version 3.8+ recommended).

Install the required dependencies:

```bash
pip install streamlit pandas numpy yfinance scikit-learn xgboost torch plotly
```

## Project Structure

```
Projects/
├── src/                # Source code
│   ├── dashboard.py    # Main dashboard application
│   └── stock_model.py  # Model logic and data fetching
├── scripts/            # Utility and test scripts
│   ├── check_indian_stocks.py
│   ├── debug_yf.py
│   └── ...
├── data/               # Data files (ignored by git)
├── docs/               # Documentation
│   ├── model_math.md   # Mathematical details
│   └── setup_steps.md  # This file
└── .gitignore
```

## Running the Dashboard

To run the main dashboard application:

1.  Navigate to the project root directory.
2.  Run the following command:

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your default web browser.

## Running Utility Scripts

You can run individual utility scripts from the `scripts/` directory:

```bash
python scripts/check_indian_stocks.py
```

## System Requirements and Portability

This project is device-agnostic and will automatically adapt to your hardware.

### Hardware Acceleration (GPU)

The Deep Learning models (LSTM/GRU) will automatically utilize your GPU if available:

*   **NVIDIA GPUs (Windows/Linux)**: Uses **CUDA** for maximum performance.
*   **Mac (M1/M2/M3)**: Uses **MPS** (Metal Performance Shaders) for high-performance training.
*   **CPU Fallback**: If no compatible GPU is found, the system defaults to **CPU**. This ensures the code runs on any standard laptop or server, though training deep learning models will be slower.

### Headless Servers (Cloud/Colab)

*   The Streamlit Dashboard (`src/dashboard.py`) functions fully on headless servers.
*   **Note**: The script `scripts/gpu_test.py` attempts to render a visual game window used for testing. This specific script requires a monitor/display and will not work in headless environments.
