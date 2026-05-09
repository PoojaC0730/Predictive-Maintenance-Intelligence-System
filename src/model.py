import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import project-specific modules
from preprocessing import load_data, create_rul, create_test_rul, drop_constant_columns, clean_data
from feature_engineering import apply_feature_engineering

def prepare_pipeline(train_file, test_file, truth_file):
    """
    Full data preparation pipeline: Load -> RUL -> Drop Constants -> Feature Engineering -> Scale.
    """
    print(f"--- Preparing Data for {os.path.basename(train_file)} ---")
    
    # 1. Load Data
    df_train = load_data(train_file)
    df_test = load_data(test_file)
    
    # 2. Create RUL Labels
    # We use a clip_rul of 125 as common practice for CMAPSS to handle the plateau in early life
    df_train = create_rul(df_train, clip_rul=125)
    df_test = create_test_rul(df_test, truth_file, clip_rul=125)
    
    # 3. Drop constant columns (sensor noise)
    df_train, df_test = drop_constant_columns(df_train, df_test)
    
    # 4. Apply Feature Engineering (Rolling stats, Trend, Health Index)
    df_train = apply_feature_engineering(df_train)
    df_test = apply_feature_engineering(df_test)
    
    # 5. Clean and Scale (StandardScaler)
    df_train_scaled, df_test_scaled = clean_data(df_train, df_test)
    
    # Define features and target
    # Exclude identifiers and the non-numeric health_status
    exclude_cols = ['unit_id', 'cycle', 'RUL', 'health_status']
    features = [col for col in df_train_scaled.columns if col not in exclude_cols]
    
    X_train = df_train_scaled[features]
    y_train = df_train_scaled['RUL']
    
    # For testing, we typically only evaluate on the LAST cycle of each unit id
    # as the ground truth RUL provided is for the end of the test sequence.
    X_test = df_test_scaled.groupby('unit_id').last()[features]
    y_test = df_test_scaled.groupby('unit_id').last()['RUL']
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Trains multiple models and compares their performance.
    """
    models = {
        "Baseline Linear Regression": LinearRegression(),
        "Robust Random Forest": RandomForestRegressor(
            n_estimators=200, 
            max_depth=15, 
            min_samples_leaf=5, 
            max_features='sqrt', 
            random_state=42, 
            n_jobs=-1
        ),
        "High-Perf XGBoost": XGBRegressor(
            n_estimators=300, 
            learning_rate=0.03, 
            max_depth=4, 
            subsample=0.8, 
            colsample_bytree=0.8, 
            random_state=42
        )
    }
    
    save_names = {
        "Baseline Linear Regression": "baseline_lr.joblib",
        "Robust Random Forest": "robust_rf.joblib",
        "High-Perf XGBoost": "high_perf_xgb.joblib"
    }
    
    results = {}
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    print("\n--- Model Training & Evaluation ---")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "MAE": round(float(mae), 4),
            "RMSE": round(float(rmse), 4),
            "R2_Score": round(float(r2), 4),
            "Model_File": save_names[name]
        }
        
        print(f"Results for {name}: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}")
        
        # Save model
        joblib.dump(model, os.path.join('models', save_names[name]))
        
    return results

if __name__ == "__main__":
    # Paths (Defaulting to FD001)
    TRAIN_PATH = 'data/raw/train_FD001.txt'
    TEST_PATH = 'data/raw/test_FD001.txt'
    TRUTH_PATH = 'data/raw/RUL_FD001.txt'
    
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Training file not found at {TRAIN_PATH}")
    else:
        # 1. Prepare Data
        X_train, y_train, X_test, y_test = prepare_pipeline(TRAIN_PATH, TEST_PATH, TRUTH_PATH)
        
        # 2. Train and Evaluate
        comparison_results = train_and_evaluate(X_train, y_train, X_test, y_test)
        
        # 3. Save Results
        os.makedirs('reports', exist_ok=True)
        with open('reports/comparison_metrics.json', 'w') as f:
            json.dump(comparison_results, f, indent=4)
            
        print("\n--- Process Complete ---")
        print("Models saved in 'models/' directory.")
        print("Metrics saved in 'reports/comparison_metrics.json'.")
