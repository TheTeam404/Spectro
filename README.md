# LIBS Software

Laser-Induced Breakdown Spectroscopy Analysis Software.

## Project Structure

*   **`frontend/`**: React User Interface
*   **`libs_engine/`**: Python Backend & Core Logic
    *   `core/`: Engine, DataManager, Exceptions
    *   `processing/`: Smoothing, Peak Detection, Peak Fitting
    *   `analysis/`: Quantification, CF-LIBS, ML Methods, Importer
    *   `external/`: NIST Fetcher
    *   `data_import/`: Raw data loading
    *   `utils/`: Helper functions
*   **`data/`**: Data storage (uploads, cache, calibration, models, samples)
*   **`api_server.py`**: Flask API definition
*   **`main.py`**: Main application entry point (runs the server)
*   **`requirements.txt`**: Python dependencies
*   **`.gitignore`**: Files ignored by Git

## Setup

1.  **Clone:** `git clone <your-repo-url> && cd LIBS_Software`
2.  **Python Env:**
    ```bash
    python -m venv .venv
    # Activate: source .venv/bin/activate (Linux/macOS) or .\.venv\Scripts\activate (Windows)
    ```
3.  **Install Python Deps:** `pip install -r requirements.txt`
4.  **Frontend Env:** `cd frontend && npm install && cd ..`
5.  **(Optional) Build Frontend:** `cd frontend && npm run build && cd ..`

## Running

1.  **Start Backend:** `python main.py` (from `LIBS_Software/` root, with venv active)
2.  **Start Frontend (Dev):** `cd frontend && npm start` (in a separate terminal)
3.  Access UI in browser (usually `http://localhost:3000`).

## Usage

(Describe how to use the software interface)