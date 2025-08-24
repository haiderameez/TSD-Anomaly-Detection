# Multivariate Time Series Anomaly Detection Using Ensemble Learning
This repository contains a Python-based machine learning solution designed to detect anomalies in multivariate time series data. It identifies data points, patterns, or events that deviate significantly from what is considered normal, helping to enable proactive and predictive maintenance strategies.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation & Execution](#installation--execution)
- [Using Your Own Data](#using-your-own-data)
- [Output Explained](#output-explained)

## Overview

The primary goal of this solution is to analyze a time series dataset, learn the patterns of "normal" behavior from a specified training period, and then score the entire dataset for abnormalities.

It is designed to detect three distinct types of anomalies:
1.  **Threshold Violations:** Variables exceeding their normal statistical ranges.
2.  **Relationship Changes:** Variables no longer following their usual correlations with other variables.
3.  **Pattern Deviations:** Temporal sequences that differ from normal operational patterns.

The final output is a copy of the input CSV file enriched with an `Abnormality_score` (from 0 to 100) and columns identifying the top 7 features contributing to each detected anomaly.

## Project Structure

The repository is organized as follows:
```bash
TSD-Anomaly-Detection/
├── data/
│ └── data.csv # Input data should be placed here
├── final_output/
│ └── ... # Stores the final CSV with anomaly scores
├── src/
│ ├── detectors/ 
│ ├── data_preprocessor.py 
│ └── unified_detector.py 
├── pipeline.py 
└──  requirements.txt 
```

## Getting Started

Follow these steps to set up and run the anomaly detection pipeline.

### Prerequisites

-   Python 3.9 or higher
-   `pip` for package installation

### Installation & Execution

1.  **Clone the Repository**
    ```powershell
    git clone https://github.com/haiderameez/TSD-Anomaly-Detection.git
    cd TSD-Anomaly-Detection
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    -   On Windows:
        ```powershell
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   On macOS/Linux:
        ```powershell
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Dependencies**
    ```python
    pip install -r requirements.txt
    ```

4.  **Place Your Data**
    Ensure your input CSV file (e.g., `data.csv`) is located inside the `/data` directory.

5.  **Run the Pipeline**
    Execute the main script from the root directory:
    ```python
    python pipeline.py
    ```

The script will print its progress to the console. Once complete, the final output file will be saved in the `/final_output` directory.

## Using Your Own Data

You can easily adapt this pipeline to run on your own dataset by following these steps:

1.  **Data Placement:**
    -   Place your custom CSV file inside the `/data` directory.

2.  **Data Format Requirements:**
    -   The file must be in **CSV format**.
    -   It must contain a `Time` column with timestamps in a standard format (e.g., `YYYY-MM-DD HH:MM:SS`).
    -   All other columns should contain **numerical data** (sensor readings, metrics, etc.).

3.  **Configure the Pipeline:**
    -   Open the `pipeline.py` file and modify the configuration constants at the top of the script:

    ```python
    # --- Configuration in pipeline.py ---

    # 1. UPDATE THE INPUT FILE PATH
    INPUT_CSV_PATH = 'data/your_custom_data.csv'

    # 2. DEFINE THE NORMAL AND ANALYSIS PERIODS FOR YOUR DATA
    TRAIN_START_TIME = 'YYYY-MM-DD HH:MM:SS'  # Start of the normal period
    TRAIN_END_TIME = 'YYYY-MM-DD HH:MM:SS'    # End of the normal period
    ANALYSIS_END_TIME = 'YYYY-MM-DD HH:MM:SS' # End of the full period to analyze
    ```

After updating these settings, run `python pipeline.py` as described above to process your custom data.


## Output Explained

The final output is a CSV file containing all the original columns plus the following 8 additional columns:

| Column Name         | Data Type | Description                                                                                             |
| ------------------- | --------- | ------------------------------------------------------------------------------------------------------- |
| `Abnormality_score` | Float     | The final anomaly score from 0.0 (normal) to 100.0 (severe).                                            |
| `top_feature_1`     | String    | The name of the feature that contributed most to the score.                                             |
| `top_feature_2`     | String    | The name of the 2nd highest contributing feature.                                                       |
| `top_feature_3`     | String    | The name of the 3rd highest contributing feature.                                                       |
| `top_feature_4`     | String    | The name of the 4th highest contributing feature.                                                       |
| `top_feature_5`     | String    | The name of the 5th highest contributing feature.                                                       |
| `top_feature_6`     | String    | The name of the 6th highest contributing feature.                                                       |
| `top_feature_7`     | String    | The name of the 7th highest contributing feature. (Empty string if fewer than 7 contributors).          |
