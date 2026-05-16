import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

# Import the pipeline from model.py to get exact features
from model import prepare_pipeline

def generate_explanations():
    print("--- Stage 5: Explainability (SHAP) ---")
    
    # 1. Prepare Data
    # Using the same data paths as model.py
    TRAIN_PATH = 'data/raw/train_FD001.txt'
    TEST_PATH = 'data/raw/test_FD001.txt'
    TRUTH_PATH = 'data/raw/RUL_FD001.txt'
    
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Data not found at {TRAIN_PATH}. Please check your paths.")
        return

    print("Loading and preprocessing data to extract test features...")
    # We use prepare_pipeline to get X_test identically to how the model was trained
    _, _, X_test, _ = prepare_pipeline(TRAIN_PATH, TEST_PATH, TRUTH_PATH)
    
    # 2. Load the Best Trained Model (XGBoost)
    model_path = 'models/high_perf_xgb.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Run model.py first.")
        return
        
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 3. Create SHAP Explainer
    print("Creating SHAP explainer and calculating SHAP values...")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    # 4. Generate Summary Plot
    print("Generating SHAP Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    
    summary_path = "reports/shap_summary.png"
    plt.savefig(summary_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved summary plot to: {summary_path}")
    
    # 5. Individual Prediction Explanation (Waterfall Plot)
    print("Generating SHAP Waterfall Plot for a single prediction...")
    plt.figure(figsize=(8, 6))
    
    # Let's explain the prediction for the first engine in the test set
    shap.plots.waterfall(shap_values[0], show=False)
    
    waterfall_path = "reports/shap_waterfall.png"
    plt.savefig(waterfall_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved waterfall plot to: {waterfall_path}")

    print("\n--- Explainability Complete ---")

if __name__ == "__main__":
    generate_explanations()
