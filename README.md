# Predictive-Maintenance-Intelligence-System

An end-to-end pipeline that ingests raw aircraft turbofan sensor data, performs exploratory data analysis, calculates Remaining Useful Life (RUL), and builds a machine learning model (using XGBoost) to predict engine failures and classify health status.

## Setup Instructions

### 1. Download the Dataset
The CMAPSS dataset is not included in this repository. Please download it and place it in the correct folder structure before running the pipeline:

1. Create a `data/raw/` directory in the root of the project.
2. Download the dataset from Kaggle: [NASA CMAPS Data](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data)
3. Extract the contents (e.g., `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) directly into the `data/raw/` folder.

### 2. Preprocessing & Data Exploration
- Run the `notebooks/data_exploration.ipynb` notebook to see the data visualizations, correlation heatmaps, and degradation trends.
- The `src/preprocessing.py` script contains the core data cleaning, feature engineering (rolling averages), and scaling logic.
