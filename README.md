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

### 3. Feature Engineering (Stage 3)
We have implemented advanced feature engineering to capture the temporal characteristics of the sensor data:
- **Rolling Statistics**: Rolling mean and standard deviation for each sensor reading over a window (default: 10 cycles) to smooth noise and capture volatility.
- **Trend Features**: Calculating the rate of change (slope) for sensor readings to detect early signs of engine degradation.
- **Health Index**: A custom health score normalized from 1.0 (perfect health) to 0.0 (failure), based on the lifecycle progress and RUL.

### ⚙️ Model Training (Stage 4)
Trained and compared three models (Linear Regression, Random Forest, XGBoost) to predict Remaining Useful Life (RUL).

### 📊 Model Performance (FD001)

| Model | MAE | RMSE | R² Score | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | 11.06 | 14.31 | 0.87 | Statistical Baseline |
| **Random Forest** | 9.37 | 12.87 | 0.90 | Robust Ensemble |
| **XGBoost** | 9.08 | 12.78 | 0.90 | High-Performance |

> [!TIP]
> **XGBoost** emerged as the best-performing model, achieving an impressive R² score of 0.90. The hyperparameters were specifically tuned (learning rate: 0.03, max depth: 4, subsample: 0.8) to prevent overfitting on noisy sensor data.

### 🧠 Explainability (Stage 5)
Integrated SHAP to provide transparency into model predictions, ensuring maintenance teams can identify the exact sensors driving failure risk.

### 🔧 Counterfactual Engine (Stage 6)
An advanced optimization engine designed to answer the core maintenance question: *"What should change to increase Remaining Useful Life (RUL)?"*
* **Sequential Sensitivity Search**: Step-by-step sequential tuning of actionable parameters (e.g. LPT speeds, HPT temperatures) to reach a target RUL.
* **Global Powell Optimization**: An L2-regularized multi-dimensional gradient-free search to find the minimum global adjustment required.
* **Domain-Specific Recommendations**: Automatic translation of numeric sensor deltas into physical maintenance tasks (e.g. blade washes, turbine cooling system checks, casing seal repairs).

*Stay tuned for more updates!*

