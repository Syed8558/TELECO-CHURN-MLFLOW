"""
MLflow Pipeline for Telco Customer Churn Prediction
5 Models: Logistic Regression, Random Forest, Decision Tree, XGBoost, CatBoost
All parameters, metrics, and results are logged to MLflow and saved as tables
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import catboost as cb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
from pathlib import Path
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "Telco_Customer_Churn_Prediction"
RESULTS_DIR = Path("mlflow_results")
RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================
def load_and_preprocess_data():
    """Load and preprocess the Telco Customer Churn dataset"""
    print("=" * 80)
    print("LOADING AND PREPROCESSING DATA")
    print("=" * 80)
    
    # Load data
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"Dataset shape: {df.shape}")
    
    # Handle TotalCharges - convert to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Drop customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Handle missing values in TotalCharges
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Convert SeniorCitizen back to numeric if needed
    if df['SeniorCitizen'].dtype == 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'No': 0, 'Yes': 1})
    
    # Separate features and target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training churn distribution: {y_train.value_counts().to_dict()}")
    print(f"Test churn distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler, categorical_cols, numerical_cols

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================
def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test, params):
    """Train a model and evaluate its performance"""
    print(f"\n{'='*80}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*80}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    cm = confusion_matrix(y_test, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1])
    }
    
    # Print metrics
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    if roc_auc:
        print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return model, metrics, y_pred, y_pred_proba

# =============================================================================
# MLFLOW LOGGING
# =============================================================================
def log_to_mlflow(model, model_name, params, metrics, X_test, y_test, y_pred, y_pred_proba):
    """Log model, parameters, and metrics to MLflow"""
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log parameters
        mlflow.log_params(params)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(metric_name, metric_value)
        
        # Log model based on type
        if model_name == "Logistic_Regression":
            mlflow.sklearn.log_model(model, "model")
        elif model_name == "Random_Forest":
            mlflow.sklearn.log_model(model, "model")
        elif model_name == "Decision_Tree":
            mlflow.sklearn.log_model(model, "model")
        elif model_name == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        elif model_name == "CatBoost":
            mlflow.catboost.log_model(model, "model")
        
        # Log additional info
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "Telco_Customer_Churn")
        mlflow.set_tag("target", "Churn")
        
        return mlflow.active_run().info.run_id

# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """Main MLflow pipeline execution"""
    print("\n" + "="*80)
    print("MLFLOW PIPELINE FOR TELCO CUSTOMER CHURN PREDICTION")
    print("="*80)
    
    # Set MLflow experiment
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"\nMLflow Experiment: {MLFLOW_EXPERIMENT_NAME}")
    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoders, scaler, categorical_cols, numerical_cols = load_and_preprocess_data()
    
    # Store all results
    all_results = []
    all_metrics = []
    all_predictions = []
    
    # =========================================================================
    # 1. LOGISTIC REGRESSION
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL 1: LOGISTIC REGRESSION")
    print("="*80)
    
    lr_params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': RANDOM_STATE
    }
    
    lr_model = LogisticRegression(**lr_params)
    lr_model, lr_metrics, lr_pred, lr_pred_proba = train_and_evaluate_model(
        lr_model, "Logistic_Regression", X_train, X_test, y_train, y_test, lr_params
    )
    
    lr_run_id = log_to_mlflow(lr_model, "Logistic_Regression", lr_params, lr_metrics, 
                              X_test, y_test, lr_pred, lr_pred_proba)
    
    all_results.append({
        'model': 'Logistic_Regression',
        'run_id': lr_run_id,
        **lr_params,
        **lr_metrics
    })
    all_metrics.append({
        'model': 'Logistic_Regression',
        'run_id': lr_run_id,
        **lr_metrics
    })
    all_predictions.append({
        'model': 'Logistic_Regression',
        'y_true': y_test.values,
        'y_pred': lr_pred,
        'y_pred_proba': lr_pred_proba if lr_pred_proba is not None else [None] * len(y_test)
    })
    
    # =========================================================================
    # 2. RANDOM FOREST
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL 2: RANDOM FOREST")
    print("="*80)
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    rf_model = RandomForestClassifier(**rf_params)
    rf_model, rf_metrics, rf_pred, rf_pred_proba = train_and_evaluate_model(
        rf_model, "Random_Forest", X_train, X_test, y_train, y_test, rf_params
    )
    
    rf_run_id = log_to_mlflow(rf_model, "Random_Forest", rf_params, rf_metrics,
                              X_test, y_test, rf_pred, rf_pred_proba)
    
    all_results.append({
        'model': 'Random_Forest',
        'run_id': rf_run_id,
        **rf_params,
        **rf_metrics
    })
    all_metrics.append({
        'model': 'Random_Forest',
        'run_id': rf_run_id,
        **rf_metrics
    })
    all_predictions.append({
        'model': 'Random_Forest',
        'y_true': y_test.values,
        'y_pred': rf_pred,
        'y_pred_proba': rf_pred_proba if rf_pred_proba is not None else [None] * len(y_test)
    })
    
    # =========================================================================
    # 3. DECISION TREE
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL 3: DECISION TREE")
    print("="*80)
    
    dt_params = {
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': RANDOM_STATE
    }
    
    dt_model = DecisionTreeClassifier(**dt_params)
    dt_model, dt_metrics, dt_pred, dt_pred_proba = train_and_evaluate_model(
        dt_model, "Decision_Tree", X_train, X_test, y_train, y_test, dt_params
    )
    
    dt_run_id = log_to_mlflow(dt_model, "Decision_Tree", dt_params, dt_metrics,
                              X_test, y_test, dt_pred, dt_pred_proba)
    
    all_results.append({
        'model': 'Decision_Tree',
        'run_id': dt_run_id,
        **dt_params,
        **dt_metrics
    })
    all_metrics.append({
        'model': 'Decision_Tree',
        'run_id': dt_run_id,
        **dt_metrics
    })
    all_predictions.append({
        'model': 'Decision_Tree',
        'y_true': y_test.values,
        'y_pred': dt_pred,
        'y_pred_proba': dt_pred_proba if dt_pred_proba is not None else [None] * len(y_test)
    })
    
    # =========================================================================
    # 4. XGBOOST
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL 4: XGBOOST")
    print("="*80)
    
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model, xgb_metrics, xgb_pred, xgb_pred_proba = train_and_evaluate_model(
        xgb_model, "XGBoost", X_train, X_test, y_train, y_test, xgb_params
    )
    
    xgb_run_id = log_to_mlflow(xgb_model, "XGBoost", xgb_params, xgb_metrics,
                               X_test, y_test, xgb_pred, xgb_pred_proba)
    
    all_results.append({
        'model': 'XGBoost',
        'run_id': xgb_run_id,
        **xgb_params,
        **xgb_metrics
    })
    all_metrics.append({
        'model': 'XGBoost',
        'run_id': xgb_run_id,
        **xgb_metrics
    })
    all_predictions.append({
        'model': 'XGBoost',
        'y_true': y_test.values,
        'y_pred': xgb_pred,
        'y_pred_proba': xgb_pred_proba if xgb_pred_proba is not None else [None] * len(y_test)
    })
    
    # =========================================================================
    # 5. CATBOOST
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL 5: CATBOOST")
    print("="*80)
    
    cat_params = {
        'iterations': 100,
        'depth': 6,
        'learning_rate': 0.1,
        'random_seed': RANDOM_STATE,
        'verbose': False,
        'loss_function': 'Logloss'
    }
    
    cat_model = cb.CatBoostClassifier(**cat_params)
    cat_model, cat_metrics, cat_pred, cat_pred_proba = train_and_evaluate_model(
        cat_model, "CatBoost", X_train, X_test, y_train, y_test, cat_params
    )
    
    cat_run_id = log_to_mlflow(cat_model, "CatBoost", cat_params, cat_metrics,
                               X_test, y_test, cat_pred, cat_pred_proba)
    
    all_results.append({
        'model': 'CatBoost',
        'run_id': cat_run_id,
        **cat_params,
        **cat_metrics
    })
    all_metrics.append({
        'model': 'CatBoost',
        'run_id': cat_run_id,
        **cat_metrics
    })
    all_predictions.append({
        'model': 'CatBoost',
        'y_true': y_test.values,
        'y_pred': cat_pred,
        'y_pred_proba': cat_pred_proba if cat_pred_proba is not None else [None] * len(y_test)
    })
    
    # =========================================================================
    # SAVE RESULTS AS TABLES
    # =========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS AS TABLES")
    print("="*80)
    
    # Convert to DataFrames
    results_df = pd.DataFrame(all_results)
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save comprehensive results table
    results_file = RESULTS_DIR / "all_models_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\n✓ Saved comprehensive results to: {results_file}")
    
    # Save metrics comparison table
    metrics_file = RESULTS_DIR / "all_models_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False)
    print(f"✓ Saved metrics comparison to: {metrics_file}")
    
    # Save predictions for each model
    for pred_data in all_predictions:
        pred_df = pd.DataFrame({
            'y_true': pred_data['y_true'],
            'y_pred': pred_data['y_pred'],
            'y_pred_proba': pred_data['y_pred_proba']
        })
        pred_file = RESULTS_DIR / f"{pred_data['model']}_predictions.csv"
        pred_df.to_csv(pred_file, index=False)
        print(f"✓ Saved {pred_data['model']} predictions to: {pred_file}")
    
    # Save parameters table
    params_list = []
    for result in all_results:
        params_dict = {'model': result['model'], 'run_id': result['run_id']}
        for key, value in result.items():
            if key not in ['model', 'run_id', 'accuracy', 'precision', 'recall', 'f1_score', 
                          'roc_auc', 'true_negatives', 'false_positives', 'false_negatives', 'true_positives']:
                params_dict[key] = value
        params_list.append(params_dict)
    
    params_df = pd.DataFrame(params_list)
    params_file = RESULTS_DIR / "all_models_parameters.csv"
    params_df.to_csv(params_file, index=False)
    print(f"✓ Saved parameters table to: {params_file}")
    
    # Save summary comparison
    summary_df = metrics_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].copy()
    summary_file = RESULTS_DIR / "models_comparison_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved comparison summary to: {summary_file}")
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print("\nMetrics Comparison:")
    print(summary_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}")
    print(f"\nTo view MLflow UI, run:")
    print(f"  mlflow ui")
    print(f"\nThen open: http://localhost:5000")
    print(f"\nExperiment Name: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Total Models Trained: {len(all_results)}")
    print("="*80)

if __name__ == "__main__":
    main()

