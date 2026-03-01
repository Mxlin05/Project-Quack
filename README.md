# Project-Quack: Intraday Trading Neural Network

[cite_start]This repository contains the codebase for a highly specialized Hybrid LSTM-Transformer neural network designed for intraday trading of the E-mini Nasdaq-100 (NQ) futures, utilizing the E-mini S&P 500 (ES) as a macroeconomic leading indicator[cite: 62]. [cite_start]The pipeline features rigorous data engineering including Fractional Differentiation, Triple Barrier Method labeling, and Combinatorial Purged Cross-Validation[cite: 63, 126, 136, 269].

## Prerequisites

Before downloading the repository, ensure you have the following system dependencies installed:

1. **Python 3.10+**: This project requires a modern Python version (development was done on Python 3.12). You can download it from [python.org](https://www.python.org/downloads/).
2. **Git Large File Storage (Git LFS)**: **Crucial Step.** This repository contains large raw `.zip` datasets that exceed standard Git limits. You *must* install Git LFS before cloning to ensure the data downloads correctly.
   * **Windows**: Download and install from [git-lfs.github.com](https://git-lfs.github.com/)
   * **Linux (Ubuntu/Debian)**: 
     ```bash
     sudo apt update
     sudo apt install git-lfs
     git lfs install
     ```
   * **Mac**: 
     ```bash
     brew install git-lfs
     git lfs install
     ```

---

## Installation & Setup

Follow these instructions from the top down to get the trading model environment running locally.

### 1. Clone the Repository
Because Git LFS is tracking the heavy `data/raw/*.zip` files, standard cloning will automatically pull the data if LFS is installed.

```bash
git clone [https://github.com/Mxlin05/Project-Quack.git](https://github.com/Mxlin05/Project-Quack.git)
cd Project-Quack
```

### 2. Create a Virtual Environment
It is highly recommended to isolate the dependencies for this project to prevent conflicts with your system Python packages.

* **Linux/Mac**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
* **Windows (Command Prompt)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

### 3. Install Required Packages
Once your virtual environment is activated (you should see `(.venv)` in your terminal prompt), install the required packages.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Folder Structure

The repository is structured to strictly separate raw data, feature processing, model architecture, and validation loops.

```text
Project-Quack/
│
├── .venv/                   # Local Python virtual environment
├── backtests/               # Storage for simulated trading run results
│
├── data/                    # Core data pipeline 
│   ├── processed/           # Stationary features 
│   ├── raw/                 # Raw 1m OHLCV data 
│   │   ├── ES_OHLCV.zip     
│   │   └── NQ_OHLCV.zip     
│   └── scalers/             # Saved Min-Max scaler objects for inference 
│
├── model/                   # Checkpoints and training artifacts 
│   ├── logs/                # TensorBoard logs and metric histories 
│   └── weights/             
│       ├── primary/         # Saved weights for the directional bias model 
│       └── secondary/       # Saved weights for the meta-labeling filter 
│
├── training/                # Core Python source code 
│   ├── architecture.py      # PyTorch classes 
│   ├── features.py          # Indicator logic, spreads, and Fractional Differentiation 
│   ├── labeling.py          # Triple Barrier Method and CUSUM filter logic 
│   └── validation.py        # Combinatorial Purged Cross-Validation logic 
│
├── .gitignore               # Ignores .venv, data/processed/, and temporary files 
├── config.yaml              # Hyperparameters 
├── main.py                  # The master execution script 
├── README.md                # Project documentation 
└── requirements.txt         # Pinned package dependencies 
```