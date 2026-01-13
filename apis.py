"""
Model APIs for Telco Customer Churn Prediction
Loads saved models from MLflow and provides prediction and confusion matrix APIs
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# =============================================================================
# CONFIGURATION
# =============================================================================
MLFLOW_EXPERIMENT_NAME = "Telco_Customer_Churn_Prediction"
RESULTS_DIR = Path(r"c:\Users\User\OneDrive\Desktop\YOLO MODELS\Hands on sssion\mlflow_results")
RANDOM_STATE = 42

# Model run IDs from saved results
MODEL_RUN_IDS = {
    'logistic_Regression': 'e760d4f743b047a4b1fda6aaf2ecf694',
    'random_Forest': '5668f3d24ddc4437a31890cf0970baa8',
    'decision_Tree': '8a85b4119d0643b1aad26e054cd6140d',
    'xGBoost': 'c54075d0946a43eda1fdf4b74feecede',
    'catBoost': '4c9d9da2dc2641f7a96ee053c94dc3bf'
}

# Global variables for models and preprocessing
models = {}
label_encoders = {}
scaler = None
categorical_cols = []
numerical_cols = []
feature_order = []

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def load_and_setup_preprocessing():
    """Load dataset and setup preprocessing (label encoders and scaler)"""
    global label_encoders, scaler, categorical_cols, numerical_cols, feature_order
    
    print("Loading dataset for preprocessing setup...")
    df = pd.read_csv('c:\\Users\\User\\OneDrive\\Desktop\\YOLO MODELS\\Hands on sssion\\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Handle TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Convert SeniorCitizen to numeric if needed
    if df['SeniorCitizen'].dtype == 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({'No': 0, 'Yes': 1})
    
    # Separate features
    X = df.drop('Churn', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create and fit label encoders
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(X[col].astype(str))
        label_encoders[col] = le
    
    # Create and fit scaler
    scaler = StandardScaler()
    scaler.fit(X[numerical_cols])
    
    # Store feature order (important for prediction)
    feature_order = X.columns.tolist()
    
    print(f"Preprocessing setup complete. Features: {len(feature_order)}")
    print(f"Categorical columns: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")

def preprocess_input(data):
    """Preprocess input data for prediction"""
    global label_encoders, scaler, categorical_cols, numerical_cols, feature_order
    
    # Convert to DataFrame if dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    
    # Handle TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        median_value = df['TotalCharges'].median()
        if pd.isna(median_value):
            median_value = 0
        df['TotalCharges'] = df['TotalCharges'].fillna(median_value)
    
    # Convert SeniorCitizen if needed
    if 'SeniorCitizen' in df.columns:
        if df['SeniorCitizen'].dtype == 'object':
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'No': 0, 'Yes': 1})
    
    # Ensure all required columns are present
    for col in feature_order:
        if col not in df.columns:
            if col in categorical_cols:
                df[col] = label_encoders[col].classes_[0]  # Use first class as default
            else:
                df[col] = 0  # Use 0 as default for numerical
    
    # Reorder columns to match training data
    df = df[feature_order]
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in df.columns:
            # Handle unseen categories
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                # If category not seen during training, use first class
                df[col] = 0
    
    # Scale numerical features
    if numerical_cols:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df.values

# =============================================================================
# MODEL LOADING
# =============================================================================
def load_models():
    """Load all models from MLflow"""
    global models
    
    print("Loading models from MLflow...")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    for model_name, run_id in MODEL_RUN_IDS.items():
        try:
            model_uri = f"runs:/{run_id}/model"
            
            # Load using appropriate flavor based on model type
            if model_name in ['Logistic_Regression', 'Random_Forest', 'Decision_Tree']:
                model = mlflow.sklearn.load_model(model_uri)
            elif model_name == 'XGBoost':
                model = mlflow.xgboost.load_model(model_uri)
            elif model_name == 'CatBoost':
                model = mlflow.catboost.load_model(model_uri)
            else:
                # Fallback to pyfunc
                model = mlflow.pyfunc.load_model(model_uri)
            
            models[model_name] = model
            print(f"✓ Loaded {model_name} (run_id: {run_id})")
        except Exception as e:
            print(f"✗ Failed to load {model_name}: {str(e)}")
            models[model_name] = None
    
    print(f"Loaded {len([m for m in models.values() if m is not None])} models successfully")

# =============================================================================
# API ROUTES
# =============================================================================
@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        "message": "Telco Customer Churn Model Prediction API",
        "endpoints": {
            "predictions": [
                "/api/predict/logistic-regression",
                "/api/predict/random-forest",
                "/api/predict/decision-tree",
                "/api/predict/xgboost",
                "/api/predict/catboost"
            ],
            "confusion_matrices": [
                "/api/confusion-matrix/logistic-regression",
                "/api/confusion-matrix/random-forest",
                "/api/confusion-matrix/decision-tree",
                "/api/confusion-matrix/xgboost",
                "/api/confusion-matrix/catboost"
            ],
            "model_info": "/api/models/info"
        }
    })

@app.route('/api/models/info', methods=['GET'])
def models_info():
    """Get information about all loaded models"""
    info = {
        "status": "success",
        "models": {}
    }
    
    for model_name, model in models.items():
        info["models"][model_name] = {
            "loaded": model is not None,
            "run_id": MODEL_RUN_IDS.get(model_name, "N/A")
        }
    
    return jsonify(info)

# =============================================================================
# PREDICTION APIs
# =============================================================================
@app.route('/api/predict/logistic-regression', methods=['POST'])
def predict_logistic_regression():
    """Predict using Logistic Regression model"""
    return predict_model('Logistic_Regression')

@app.route('/api/predict/random-forest', methods=['POST'])
def predict_random_forest():
    """Predict using Random Forest model"""
    return predict_model('Random_Forest')

@app.route('/api/predict/decision-tree', methods=['POST'])
def predict_decision_tree():
    """Predict using Decision Tree model"""
    return predict_model('Decision_Tree')

@app.route('/api/predict/xgboost', methods=['POST'])
def predict_xgboost():
    """Predict using XGBoost model"""
    return predict_model('XGBoost')

@app.route('/api/predict/catboost', methods=['POST'])
def predict_catboost():
    """Predict using CatBoost model"""
    return predict_model('CatBoost')

def predict_model(model_name):
    """Generic prediction function"""
    if model_name not in models or models[model_name] is None:
        return jsonify({
            "status": "error",
            "message": f"Model {model_name} not loaded"
        }), 404
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided. Please send JSON data in request body."
            }), 400
        
        # Preprocess input
        X_processed = preprocess_input(data)
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict(X_processed)
        prediction_proba = None
        
        # Try to get prediction probabilities
        try:
            if hasattr(model, 'predict_proba'):
                prediction_proba = model.predict_proba(X_processed)
        except Exception as e:
            # If predict_proba fails, continue without probabilities
            pass
        
        # Format results
        result = {
            "status": "success",
            "model": model_name,
            "prediction": int(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
            "prediction_label": "Churn" if prediction[0] == 1 else "No Churn"
        }
        
        if prediction_proba is not None:
            if len(prediction_proba.shape) == 2:
                result["prediction_probability"] = {
                    "no_churn": float(prediction_proba[0][0]),
                    "churn": float(prediction_proba[0][1])
                }
            else:
                result["prediction_probability"] = float(prediction_proba[0])
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =============================================================================
# CONFUSION MATRIX APIs
# =============================================================================
@app.route('/api/confusion-matrix/logistic-regression', methods=['GET'])
def confusion_matrix_logistic_regression():
    """Get confusion matrix for Logistic Regression model"""
    return get_confusion_matrix('Logistic_Regression')

@app.route('/api/confusion-matrix/random-forest', methods=['GET'])
def confusion_matrix_random_forest():
    """Get confusion matrix for Random Forest model"""
    return get_confusion_matrix('Random_Forest')

@app.route('/api/confusion-matrix/decision-tree', methods=['GET'])
def confusion_matrix_decision_tree():
    """Get confusion matrix for Decision Tree model"""
    return get_confusion_matrix('Decision_Tree')

@app.route('/api/confusion-matrix/xgboost', methods=['GET'])
def confusion_matrix_xgboost():
    """Get confusion matrix for XGBoost model"""
    return get_confusion_matrix('XGBoost')

@app.route('/api/confusion-matrix/catboost', methods=['GET'])
def confusion_matrix_catboost():
    """Get confusion matrix for CatBoost model"""
    return get_confusion_matrix('CatBoost')

def get_confusion_matrix(model_name):
    """Get confusion matrix from saved predictions"""
    try:
        # Load saved predictions
        pred_file = RESULTS_DIR / f"{model_name}_predictions.csv"
        
        if not pred_file.exists():
            return jsonify({
                "status": "error",
                "message": f"Prediction file not found for {model_name}"
            }), 404
        
        # Read predictions
        pred_df = pd.read_csv(pred_file)
        
        if 'y_true' not in pred_df.columns or 'y_pred' not in pred_df.columns:
            return jsonify({
                "status": "error",
                "message": "Invalid prediction file format"
            }), 400
        
        y_true = pred_df['y_true'].values
        y_pred = pred_df['y_pred'].values
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Format response
        result = {
            "status": "success",
            "model": model_name,
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "matrix": [
                    [int(tn), int(fp)],
                    [int(fn), int(tp)]
                ],
                "labels": ["No Churn", "Churn"]
            },
            "metrics": {
                "accuracy": round(accuracy, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            },
            "summary": {
                "total_samples": int(len(y_true)),
                "correct_predictions": int(tp + tn),
                "incorrect_predictions": int(fp + fn)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =============================================================================
# BATCH PREDICTION API
# =============================================================================
@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction using all models"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "status": "error",
                "message": "No data provided"
            }), 400
        
        # Ensure data is a list
        if not isinstance(data, list):
            data = [data]
        
        # Preprocess all inputs
        X_processed = preprocess_input(data)
        
        results = {
            "status": "success",
            "predictions": []
        }
        
        # Get predictions from all models
        for model_name, model in models.items():
            if model is None:
                continue
            
            try:
                predictions = model.predict(X_processed)
                
                model_results = {
                    "model": model_name,
                    "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions),
                    "predictions_labels": ["Churn" if p == 1 else "No Churn" for p in predictions]
                }
                
                # Try to get probabilities
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_processed)
                        if len(proba.shape) == 2 and proba.shape[1] == 2:
                            model_results["probabilities"] = [
                                {"no_churn": float(p[0]), "churn": float(p[1])} for p in proba
                            ]
                except Exception as e:
                    pass
                
                results["predictions"].append(model_results)
            
            except Exception as e:
                results["predictions"].append({
                    "model": model_name,
                    "error": str(e)
                })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# =============================================================================
# INITIALIZATION
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("INITIALIZING MODEL APIs")
    print("=" * 80)
    
    # Setup preprocessing
    load_and_setup_preprocessing()
    
    # Load models
    load_models()
    
    print("\n" + "=" * 80)
    print("MODEL APIs READY")
    print("=" * 80)
    print("API available at http://localhost:5001")
    print("\nEndpoints:")
    print("  - POST /api/predict/logistic-regression")
    print("  - POST /api/predict/random-forest")
    print("  - POST /api/predict/decision-tree")
    print("  - POST /api/predict/xgboost")
    print("  - POST /api/predict/catboost")
    print("  - GET  /api/confusion-matrix/logistic-regression")
    print("  - GET  /api/confusion-matrix/random-forest")
    print("  - GET  /api/confusion-matrix/decision-tree")
    print("  - GET  /api/confusion-matrix/xgboost")
    print("  - GET  /api/confusion-matrix/catboost")
    print("  - POST /api/predict/batch (predict with all models)")
    print("  - GET  /api/models/info")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5001)

