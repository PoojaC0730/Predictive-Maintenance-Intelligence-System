import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Add src folder to path to ensure all imports work correctly
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import preparation pipeline
from model import prepare_pipeline

def _get_feature_dependencies(feature_names):
    """
    Maps each base feature (e.g., 'sensor_2') to its dependent features
    (e.g., 'sensor_2_roll_mean', 'sensor_2_trend') present in the feature set.
    """
    dependencies = {}
    base_features = []
    
    # Identify non-derived base features
    for col in feature_names:
        if not col.endswith('_roll_mean') and not col.endswith('_roll_std') and not col.endswith('_trend'):
            base_features.append(col)
            
    for base in base_features:
        deps = []
        for col in feature_names:
            if col == base:
                continue
            if col.startswith(base + '_'):
                deps.append(col)
        # Even if a base feature has no explicit rolling dependents, we track it
        dependencies[base] = deps
            
    return dependencies

def apply_perturbation(instance, base_feature, delta, dependencies, window=10):
    """
    Applies a perturbation delta to a base feature and propagates the change
    to all its dependent rolling/trend features.
    """
    perturbed = instance.copy()
    if base_feature in perturbed.index:
        perturbed[base_feature] += delta
        
    for dep in dependencies.get(base_feature, []):
        if dep.endswith('_roll_mean'):
            perturbed[dep] += delta / window
        elif dep.endswith('_trend'):
            perturbed[dep] += delta / window
        # We leave _roll_std unchanged as it is non-linear and less sensitive to a single-step change
        
    return perturbed

class CounterfactualEngine:
    """
    State-of-the-art Counterfactual Engine for Predictive Maintenance.
    Answers: "What should change to increase Remaining Useful Life (RUL)?"
    """
    def __init__(self, model, feature_names, X_train=None, window=10):
        """
        Args:
            model: Trained machine learning model (e.g. XGBRegressor).
            feature_names (list): List of feature names used by the model.
            X_train (pd.DataFrame, optional): Training feature matrix to extract bounds from.
            window (int): Rolling window size used in feature engineering.
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.window = window
        self.dependencies = _get_feature_dependencies(self.feature_names)
        
        # Establish feature bounds to ensure recommended changes are physically realistic
        self.bounds = {}
        if X_train is not None:
            for col in self.feature_names:
                self.bounds[col] = (X_train[col].min(), X_train[col].max())
        else:
            # Fallback to standard normal bounds in scaled space
            for col in self.feature_names:
                self.bounds[col] = (-3.0, 3.0)
                
    def explain_sequential(self, instance, target_rul_increase, actionable_features=None, max_steps=30, step_size=0.05):
        """
        Finds a counterfactual state by sequentially perturbing the most sensitive actionable features.
        This simulates step-by-step sequential maintenance actions.
        """
        if actionable_features is None:
            # Default to base features (sensors/settings) present in the dependencies
            actionable_features = [f for f in self.dependencies.keys() if 'sensor' in f or 'setting' in f]
            
        current_state = instance.copy()
        initial_pred = float(self.model.predict(pd.DataFrame([current_state])[self.feature_names])[0])
        target_pred = initial_pred + target_rul_increase
        
        steps = []
        current_pred = initial_pred
        
        for step in range(max_steps):
            if current_pred >= target_pred:
                break
                
            best_feature = None
            best_delta = 0
            best_improvement = -float('inf')
            
            # Test positive and negative steps for all actionable features
            for feature in actionable_features:
                for direction in [1, -1]:
                    delta = direction * step_size
                    
                    # Check bounds before applying
                    current_val = current_state[feature]
                    new_val = current_val + delta
                    min_b, max_b = self.bounds.get(feature, (-3.0, 3.0))
                    if not (min_b <= new_val <= max_b):
                        continue
                        
                    temp_state = apply_perturbation(current_state, feature, delta, self.dependencies, self.window)
                    temp_pred = float(self.model.predict(pd.DataFrame([temp_state])[self.feature_names])[0])
                    
                    improvement = temp_pred - current_pred
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_feature = feature
                        best_delta = delta
                        
            # If no step gives any improvement, stop search
            if best_improvement <= 0 or best_feature is None:
                break
                
            # Apply the best step
            current_state = apply_perturbation(current_state, best_feature, best_delta, self.dependencies, self.window)
            current_pred = float(self.model.predict(pd.DataFrame([current_state])[self.feature_names])[0])
            
            steps.append({
                'step': step + 1,
                'feature': best_feature,
                'change': best_delta,
                'new_prediction': round(current_pred, 4),
                'improvement': round(best_improvement, 4)
            })
            
        # Calculate aggregate modifications
        modifications = {}
        for feature in actionable_features:
            initial_val = instance[feature]
            recommended_val = current_state[feature]
            change = recommended_val - initial_val
            if abs(change) > 1e-5:
                modifications[feature] = {
                    'initial': round(float(initial_val), 4),
                    'recommended': round(float(recommended_val), 4),
                    'change': round(float(change), 4)
                }
                
        success = current_pred >= target_pred
        
        return {
            'method': 'Sequential Sensitivity Search',
            'success': success,
            'initial_prediction': round(initial_pred, 4),
            'target_prediction': round(target_pred, 4),
            'achieved_prediction': round(current_pred, 4),
            'modifications': modifications,
            'steps': steps
        }

    def explain_optimization(self, instance, target_rul_increase, actionable_features=None, l2_penalty=0.1):
        """
        Finds an optimal counterfactual state by solving a bound-constrained
        optimization problem using SciPy's gradient-free Powell solver.
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            # Fallback if scipy is not installed
            return {
                'method': 'Global Powell Optimization',
                'success': False,
                'error': 'SciPy is required for global optimization method. Please install using: pip install scipy',
                'initial_prediction': round(float(self.model.predict(pd.DataFrame([instance])[self.feature_names])[0]), 4),
                'target_prediction': round(float(self.model.predict(pd.DataFrame([instance])[self.feature_names])[0]) + target_rul_increase, 4),
                'achieved_prediction': round(float(self.model.predict(pd.DataFrame([instance])[self.feature_names])[0]), 4),
                'modifications': {}
            }
            
        if actionable_features is None:
            actionable_features = [f for f in self.dependencies.keys() if 'sensor' in f or 'setting' in f]
            
        initial_pred = float(self.model.predict(pd.DataFrame([instance])[self.feature_names])[0])
        target_pred = initial_pred + target_rul_increase
        
        n_features = len(actionable_features)
        initial_deltas = np.zeros(n_features)
        
        # Define bounds for delta based on feature bounds
        bounds = []
        for f in actionable_features:
            current_val = instance[f]
            min_b, max_b = self.bounds.get(f, (-3.0, 3.0))
            bounds.append((min_b - current_val, max_b - current_val))
            
        def objective(deltas):
            perturbed_state = instance.copy()
            for idx, feature in enumerate(actionable_features):
                delta = deltas[idx]
                perturbed_state = apply_perturbation(perturbed_state, feature, delta, self.dependencies, self.window)
                
            pred = float(self.model.predict(pd.DataFrame([perturbed_state])[self.feature_names])[0])
            
            # Penalize missing target and penalize large alterations (L2 regularization)
            prediction_gap = max(0, target_pred - pred)
            distance_penalty = np.sum(deltas ** 2)
            
            return (prediction_gap ** 2) + l2_penalty * distance_penalty
            
        res = minimize(objective, initial_deltas, method='Powell', bounds=bounds)
        
        recommended_state = instance.copy()
        modifications = {}
        for idx, feature in enumerate(actionable_features):
            delta = res.x[idx]
            if abs(delta) > 1e-4:
                recommended_state = apply_perturbation(recommended_state, feature, delta, self.dependencies, self.window)
                modifications[feature] = {
                    'initial': round(float(instance[feature]), 4),
                    'recommended': round(float(recommended_state[feature]), 4),
                    'change': round(float(delta), 4)
                }
                
        final_pred = float(self.model.predict(pd.DataFrame([recommended_state])[self.feature_names])[0])
        success = final_pred >= target_pred
        
        return {
            'method': 'Global Powell Optimization',
            'success': success,
            'initial_prediction': round(initial_pred, 4),
            'target_prediction': round(target_pred, 4),
            'achieved_prediction': round(final_pred, 4),
            'modifications': modifications
        }

def generate_recommendations_text(result):
    """
    Generates structured, human-readable actionable maintenance recommendations.
    """
    if 'error' in result:
        return f"Error running method: {result['error']}"
        
    recs = []
    method = result['method']
    success = result['success']
    initial = result['initial_prediction']
    achieved = result['achieved_prediction']
    target = result['target_prediction']
    
    recs.append(f"=== COUNTERFACTUAL ANALYSIS REPORT ({method.upper()}) ===")
    recs.append(f"Current Predicted RUL  : {initial:.2f} cycles")
    recs.append(f"Target RUL (Objective) : {target:.2f} cycles")
    recs.append(f"Achieved RUL           : {achieved:.2f} cycles")
    recs.append(f"Status                 : {'SUCCESS ✅' if success else 'PARTIAL SUCCESS ⚠️'}")
    recs.append("-" * 55)
    
    if not result['modifications']:
        recs.append("No modifications were required or found to increase RUL.")
        return "\n".join(recs)
        
    recs.append("Recommended Parametric Adjustments:")
    for feature, info in result['modifications'].items():
        change = info['change']
        direction = "increase" if change > 0 else "decrease"
        symbol = "↑" if change > 0 else "↓"
        recs.append(f"  * {symbol} {feature.upper()}: {direction.capitalize()} reading by {abs(change):.4f} units (from {info['initial']:.4f} to {info['recommended']:.4f})")
        
    # Translate to domain-specific context
    recs.append("\nSuggested Maintenance Interpretations:")
    sensor_descriptions = {
        'sensor_2': 'LPT (Low Pressure Turbine) Speed',
        'sensor_3': 'HPT (High Pressure Turbine) Speed',
        'sensor_4': 'LPT Outlet Temp',
        'sensor_7': 'Ratio of Fuel Flow to HP Compressor Outlet Pressure',
        'sensor_8': 'Bypass Ratio',
        'sensor_11': 'HPT Outlet Temp',
        'sensor_12': 'LPT Outlet Pressure',
        'sensor_13': 'Engine Pressure Ratio',
        'sensor_14': 'HP Compressor Speed',
        'sensor_15': 'LPT Outlet Temp (Auxiliary)',
        'sensor_17': 'HPT Speed',
        'sensor_20': 'HPT Outlet Temp (Channel 2)',
        'sensor_21': 'LPT Outlet Temp (Channel 2)'
    }
    
    for feature, info in result['modifications'].items():
        change = info['change']
        desc = sensor_descriptions.get(feature, 'Operational setting / Sensor reading')
        
        # Rule-based domain interpretation of CMAPSS features
        if 'Temp' in desc:
            action = "Improve cooling air flow, clean turbine blade thermal barrier coatings, or check for core gas path degradation to lower temperatures" if change < 0 else "Inspect temperature sensor calibration or burner nozzles"
        elif 'Bypass' in desc:
            action = "Check bypass duct, adjust fan blade angles, or inspect guide vanes to optimize bypass ratio"
        elif 'Speed' in desc:
            action = "Perform turbine wash, balance rotors, or clean compressor blades to restore efficient compressor/turbine speeds" if change > 0 else "Check engine governor and fuel flow settings to prevent overspeed"
        elif 'Pressure' in desc or 'Ratio' in desc:
            action = "Seal leakage paths in pressure casings, clean compressor seals, or tune variable stator vanes to restore pressure profile" if change > 0 else "Inspect discharge valves and bleed air systems"
        else:
            action = f"Perform general inspection and calibration of the {desc} subsystem"
            
        recs.append(f"  - {feature.upper()} ({desc}):\n    👉 Recommended Action: {action}")
        
    return "\n".join(recs)

if __name__ == "__main__":
    print("--- Stage 6: Counterfactual Engine (Actionable Insights) ---")
    
    # 1. Prepare Data using model pipeline
    TRAIN_PATH = 'data/raw/train_FD001.txt'
    TEST_PATH = 'data/raw/test_FD001.txt'
    TRUTH_PATH = 'data/raw/RUL_FD001.txt'
    
    if not os.path.exists(TRAIN_PATH):
        print(f"Error: Data not found at {TRAIN_PATH}. Please ensure Kaggle dataset is downloaded.")
        sys.exit(1)
        
    print("Loading data and extracting features...")
    X_train, y_train, X_test, y_test = prepare_pipeline(TRAIN_PATH, TEST_PATH, TRUTH_PATH)
    
    # 2. Load the best model (XGBoost)
    model_path = 'models/high_perf_xgb.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}. Please run model.py first.")
        sys.exit(1)
        
    print(f"Loading best-performing model from: {model_path}")
    model = joblib.load(model_path)
    
    # 3. Instantiate the Counterfactual Engine
    engine = CounterfactualEngine(model=model, feature_names=X_train.columns, X_train=X_train)
    
    # 4. Pick an engine that is critical / warning status for counterfactual explanation
    # Let's find an engine with low predicted RUL so that actionable insights are highly useful
    preds = model.predict(X_test)
    test_results = pd.DataFrame({'unit_id': X_test.index, 'actual_rul': y_test.values, 'predicted_rul': preds})
    
    # Find critical engine (e.g., predicted RUL < 50)
    critical_engines = test_results[test_results['predicted_rul'] < 50].sort_values(by='predicted_rul')
    
    if len(critical_engines) == 0:
        # Fallback to the engine with the lowest predicted RUL
        critical_engines = test_results.sort_values(by='predicted_rul')
        
    selected_unit = int(critical_engines.iloc[0]['unit_id'])
    initial_rul = critical_engines.iloc[0]['predicted_rul']
    actual_rul = critical_engines.iloc[0]['actual_rul']
    
    print(f"\nTargeting Engine Unit ID: {selected_unit}")
    print(f"Current Predicted RUL  : {initial_rul:.2f} cycles")
    print(f"Actual Remaining RUL   : {actual_rul:.2f} cycles")
    
    # Extract the feature vector for this engine unit
    instance = X_test.loc[selected_unit]
    
    # Target RUL increase (let's try to increase RUL by 25 cycles or up to 30 cycles)
    target_increase = 30.0
    print(f"Objective: Recommend maintenance actions to increase predicted RUL by +{target_increase} cycles.")
    
    # 5. Run Sequential Sensitivity Method
    print("\n[Method 1] Running Sequential Sensitivity Search...")
    seq_result = engine.explain_sequential(instance, target_rul_increase=target_increase, step_size=0.1)
    seq_report = generate_recommendations_text(seq_result)
    print(seq_report)
    
    # 6. Run Global Powell Optimization Method
    print("\n[Method 2] Running Global Powell Optimization...")
    opt_result = engine.explain_optimization(instance, target_rul_increase=target_increase, l2_penalty=0.5)
    opt_report = generate_recommendations_text(opt_result)
    print(opt_report)
    
    # 7. Save results to reports directory
    os.makedirs('reports', exist_ok=True)
    report_data = {
        'unit_id': selected_unit,
        'current_predicted_rul': round(float(initial_rul), 4),
        'actual_remaining_rul': round(float(actual_rul), 4),
        'target_increase': target_increase,
        'methods': {
            'sequential_search': seq_result,
            'powell_optimization': opt_result
        }
    }
    
    report_path = 'reports/counterfactual_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)
        
    print(f"\nSuccessfully saved full counterfactual analysis report to: {report_path}")
