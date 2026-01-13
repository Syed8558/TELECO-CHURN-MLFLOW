"""
Flask API for Telco Customer Churn EDA Graphs
Returns JSON data for all visualizations
"""

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load dataset once at startup
print("Loading dataset...")
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle TotalCharges - it might have empty strings
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Convert SeniorCitizen to string for consistency
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

@app.route('/')
def index():
    """API root endpoint"""
    return jsonify({
        "message": "Telco Customer Churn EDA API",
        "endpoints": [
            "/api/churn-distribution",
            "/api/churn-count",
            "/api/gender-churn",
            "/api/senior-citizen-churn",
            "/api/partner-dependents",
            "/api/tenure-distribution",
            "/api/monthly-charges",
            "/api/total-charges",
            "/api/contract-churn",
            "/api/payment-method-churn",
            "/api/internet-service-churn",
            "/api/service-features",
            "/api/correlation-heatmap",
            "/api/tenure-vs-monthly-charges",
            "/api/churn-rates",
            "/api/statistical-summary",
            "/api/numeric-distributions"
        ]
    })

@app.route('/api/churn-distribution', methods=['GET'])
def churn_distribution():
    """1. Churn Distribution (Pie Chart)"""
    churn_counts = df['Churn'].value_counts()
    return jsonify({
        "chart_type": "pie",
        "title": "Customer Churn Distribution",
        "data": {
            "labels": churn_counts.index.tolist(),
            "values": churn_counts.values.tolist(),
            "percentages": [round((v / len(df)) * 100, 2) for v in churn_counts.values]
        }
    })

@app.route('/api/churn-count', methods=['GET'])
def churn_count():
    """2. Churn Count Bar Plot"""
    churn_counts = df['Churn'].value_counts()
    return jsonify({
        "chart_type": "bar",
        "title": "Customer Churn Count",
        "data": {
            "labels": churn_counts.index.tolist(),
            "values": churn_counts.values.tolist()
        }
    })

@app.route('/api/gender-churn', methods=['GET'])
def gender_churn():
    """3. Gender Distribution by Churn"""
    churn_gender = pd.crosstab(df['gender'], df['Churn'])
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Gender Distribution by Churn Status",
        "data": {
            "categories": churn_gender.index.tolist(),
            "series": [
                {
                    "name": "No",
                    "data": churn_gender['No'].tolist() if 'No' in churn_gender.columns else []
                },
                {
                    "name": "Yes",
                    "data": churn_gender['Yes'].tolist() if 'Yes' in churn_gender.columns else []
                }
            ]
        }
    })

@app.route('/api/senior-citizen-churn', methods=['GET'])
def senior_citizen_churn():
    """4. Senior Citizen Distribution by Churn"""
    senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'])
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Senior Citizen Distribution by Churn Status",
        "data": {
            "categories": senior_churn.index.tolist(),
            "series": [
                {
                    "name": "No",
                    "data": senior_churn['No'].tolist() if 'No' in senior_churn.columns else []
                },
                {
                    "name": "Yes",
                    "data": senior_churn['Yes'].tolist() if 'Yes' in senior_churn.columns else []
                }
            ]
        }
    })

@app.route('/api/partner-dependents', methods=['GET'])
def partner_dependents():
    """5. Partner and Dependents Analysis"""
    partner_churn = pd.crosstab(df['Partner'], df['Churn'])
    dependents_churn = pd.crosstab(df['Dependents'], df['Churn'])
    
    return jsonify({
        "chart_type": "multi_grouped_bar",
        "title": "Partner & Dependents Status by Churn",
        "data": {
            "partner": {
                "categories": partner_churn.index.tolist(),
                "series": [
                    {
                        "name": "No",
                        "data": partner_churn['No'].tolist() if 'No' in partner_churn.columns else []
                    },
                    {
                        "name": "Yes",
                        "data": partner_churn['Yes'].tolist() if 'Yes' in partner_churn.columns else []
                    }
                ]
            },
            "dependents": {
                "categories": dependents_churn.index.tolist(),
                "series": [
                    {
                        "name": "No",
                        "data": dependents_churn['No'].tolist() if 'No' in dependents_churn.columns else []
                    },
                    {
                        "name": "Yes",
                        "data": dependents_churn['Yes'].tolist() if 'Yes' in dependents_churn.columns else []
                    }
                ]
            }
        }
    })

@app.route('/api/tenure-distribution', methods=['GET'])
def tenure_distribution():
    """6. Tenure Distribution"""
    # Histogram data
    tenure_hist, bins = np.histogram(df['tenure'].dropna(), bins=50)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    
    # Box plot data by churn
    churn_no_tenure = df[df['Churn'] == 'No']['tenure'].dropna().tolist()
    churn_yes_tenure = df[df['Churn'] == 'Yes']['tenure'].dropna().tolist()
    
    return jsonify({
        "chart_type": "multi_chart",
        "title": "Tenure Distribution",
        "data": {
            "histogram": {
                "x": bin_centers,
                "y": tenure_hist.tolist(),
                "xlabel": "Tenure (months)",
                "ylabel": "Frequency"
            },
            "boxplot": {
                "labels": ["No Churn", "Churn"],
                "data": [churn_no_tenure, churn_yes_tenure]
            }
        }
    })

@app.route('/api/monthly-charges', methods=['GET'])
def monthly_charges():
    """7. Monthly Charges Distribution"""
    # Histogram data
    charges_hist, bins = np.histogram(df['MonthlyCharges'].dropna(), bins=50)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    
    # Box plot data by churn
    churn_no_charges = df[df['Churn'] == 'No']['MonthlyCharges'].dropna().tolist()
    churn_yes_charges = df[df['Churn'] == 'Yes']['MonthlyCharges'].dropna().tolist()
    
    return jsonify({
        "chart_type": "multi_chart",
        "title": "Monthly Charges Distribution",
        "data": {
            "histogram": {
                "x": bin_centers,
                "y": charges_hist.tolist(),
                "xlabel": "Monthly Charges ($)",
                "ylabel": "Frequency"
            },
            "boxplot": {
                "labels": ["No Churn", "Churn"],
                "data": [churn_no_charges, churn_yes_charges]
            }
        }
    })

@app.route('/api/total-charges', methods=['GET'])
def total_charges():
    """8. Total Charges Distribution"""
    total_charges_clean = df['TotalCharges'].dropna()
    
    # Histogram data
    charges_hist, bins = np.histogram(total_charges_clean, bins=50)
    bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    
    # Box plot data by churn
    df_clean = df.dropna(subset=['TotalCharges'])
    churn_no_charges = df_clean[df_clean['Churn'] == 'No']['TotalCharges'].tolist()
    churn_yes_charges = df_clean[df_clean['Churn'] == 'Yes']['TotalCharges'].tolist()
    
    return jsonify({
        "chart_type": "multi_chart",
        "title": "Total Charges Distribution",
        "data": {
            "histogram": {
                "x": bin_centers,
                "y": charges_hist.tolist(),
                "xlabel": "Total Charges ($)",
                "ylabel": "Frequency"
            },
            "boxplot": {
                "labels": ["No Churn", "Churn"],
                "data": [churn_no_charges, churn_yes_charges]
            }
        }
    })

@app.route('/api/contract-churn', methods=['GET'])
def contract_churn():
    """9. Contract Type Analysis"""
    contract_churn = pd.crosstab(df['Contract'], df['Churn'])
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Contract Type Distribution by Churn Status",
        "data": {
            "categories": contract_churn.index.tolist(),
            "series": [
                {
                    "name": "No",
                    "data": contract_churn['No'].tolist() if 'No' in contract_churn.columns else []
                },
                {
                    "name": "Yes",
                    "data": contract_churn['Yes'].tolist() if 'Yes' in contract_churn.columns else []
                }
            ]
        }
    })

@app.route('/api/payment-method-churn', methods=['GET'])
def payment_method_churn():
    """10. Payment Method Analysis"""
    payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'])
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Payment Method Distribution by Churn Status",
        "data": {
            "categories": payment_churn.index.tolist(),
            "series": [
                {
                    "name": "No",
                    "data": payment_churn['No'].tolist() if 'No' in payment_churn.columns else []
                },
                {
                    "name": "Yes",
                    "data": payment_churn['Yes'].tolist() if 'Yes' in payment_churn.columns else []
                }
            ]
        }
    })

@app.route('/api/internet-service-churn', methods=['GET'])
def internet_service_churn():
    """11. Internet Service Analysis"""
    internet_churn = pd.crosstab(df['InternetService'], df['Churn'])
    return jsonify({
        "chart_type": "grouped_bar",
        "title": "Internet Service Type Distribution by Churn Status",
        "data": {
            "categories": internet_churn.index.tolist(),
            "series": [
                {
                    "name": "No",
                    "data": internet_churn['No'].tolist() if 'No' in internet_churn.columns else []
                },
                {
                    "name": "Yes",
                    "data": internet_churn['Yes'].tolist() if 'Yes' in internet_churn.columns else []
                }
            ]
        }
    })

@app.route('/api/service-features', methods=['GET'])
def service_features():
    """12. Service Features Analysis"""
    service_features = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'PaperlessBilling']
    
    features_data = {}
    for feature in service_features:
        feature_churn = pd.crosstab(df[feature], df['Churn'])
        features_data[feature] = {
            "categories": feature_churn.index.tolist(),
            "series": [
                {
                    "name": "No",
                    "data": feature_churn['No'].tolist() if 'No' in feature_churn.columns else []
                },
                {
                    "name": "Yes",
                    "data": feature_churn['Yes'].tolist() if 'Yes' in feature_churn.columns else []
                }
            ]
        }
    
    return jsonify({
        "chart_type": "multi_grouped_bar",
        "title": "Service Features by Churn",
        "data": features_data
    })

@app.route('/api/correlation-heatmap', methods=['GET'])
def correlation_heatmap():
    """13. Correlation Heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'customerID' in numeric_cols:
        numeric_cols.remove('customerID')
    
    corr_matrix = df[numeric_cols].corr()
    
    return jsonify({
        "chart_type": "heatmap",
        "title": "Correlation Heatmap of Numeric Features",
        "data": {
            "labels": corr_matrix.columns.tolist(),
            "values": corr_matrix.values.tolist(),
            "matrix": corr_matrix.to_dict()
        }
    })

@app.route('/api/tenure-vs-monthly-charges', methods=['GET'])
def tenure_vs_monthly_charges():
    """14. Tenure vs Monthly Charges Scatter Plot"""
    churn_yes = df[df['Churn'] == 'Yes']
    churn_no = df[df['Churn'] == 'No']
    
    return jsonify({
        "chart_type": "scatter",
        "title": "Tenure vs Monthly Charges by Churn Status",
        "data": {
            "series": [
                {
                    "name": "No Churn",
                    "x": churn_no['tenure'].tolist(),
                    "y": churn_no['MonthlyCharges'].tolist()
                },
                {
                    "name": "Churn",
                    "x": churn_yes['tenure'].tolist(),
                    "y": churn_yes['MonthlyCharges'].tolist()
                }
            ],
            "xlabel": "Tenure (months)",
            "ylabel": "Monthly Charges ($)"
        }
    })

@app.route('/api/churn-rates', methods=['GET'])
def churn_rates():
    """15. Churn Rate by Contract and Payment Method"""
    # Contract churn rate
    contract_churn_rate = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    
    # Payment Method churn rate
    payment_churn_rate = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100)
    
    return jsonify({
        "chart_type": "multi_bar",
        "title": "Churn Rates",
        "data": {
            "contract": {
                "labels": contract_churn_rate.index.tolist(),
                "values": [round(v, 2) for v in contract_churn_rate.values.tolist()],
                "ylabel": "Churn Rate (%)"
            },
            "payment_method": {
                "labels": payment_churn_rate.index.tolist(),
                "values": [round(v, 2) for v in payment_churn_rate.values.tolist()],
                "ylabel": "Churn Rate (%)"
            }
        }
    })

@app.route('/api/statistical-summary', methods=['GET'])
def statistical_summary():
    """16. Statistical Summary by Churn"""
    metrics = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    summary_data = {}
    for metric in metrics:
        churn_yes_mean = df[df['Churn'] == 'Yes'][metric].mean()
        churn_no_mean = df[df['Churn'] == 'No'][metric].mean()
        
        summary_data[metric] = {
            "labels": ["No Churn", "Churn"],
            "values": [
                round(churn_no_mean, 2) if not pd.isna(churn_no_mean) else None,
                round(churn_yes_mean, 2) if not pd.isna(churn_yes_mean) else None
            ]
        }
    
    return jsonify({
        "chart_type": "multi_bar",
        "title": "Statistical Summary by Churn",
        "data": summary_data
    })

@app.route('/api/numeric-distributions', methods=['GET'])
def numeric_distributions():
    """17. Distribution of all numeric features"""
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    distributions = {}
    for feature in numeric_features:
        churn_yes = df[df['Churn'] == 'Yes'][feature].dropna()
        churn_no = df[df['Churn'] == 'No'][feature].dropna()
        
        # Create histogram data
        hist_yes, bins_yes = np.histogram(churn_yes, bins=30)
        hist_no, bins_no = np.histogram(churn_no, bins=30)
        
        bin_centers_yes = [(bins_yes[i] + bins_yes[i+1]) / 2 for i in range(len(bins_yes)-1)]
        bin_centers_no = [(bins_no[i] + bins_no[i+1]) / 2 for i in range(len(bins_no)-1)]
        
        distributions[feature] = {
            "series": [
                {
                    "name": "No Churn",
                    "x": bin_centers_no,
                    "y": hist_no.tolist()
                },
                {
                    "name": "Churn",
                    "x": bin_centers_yes,
                    "y": hist_yes.tolist()
                }
            ],
            "xlabel": feature,
            "ylabel": "Frequency"
        }
    
    return jsonify({
        "chart_type": "multi_histogram",
        "title": "Numeric Features Distribution",
        "data": distributions
    })

@app.route('/api/dataset-info', methods=['GET'])
def dataset_info():
    """Get basic dataset information"""
    return jsonify({
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1])
        },
        "churn_distribution": {
            "no_churn": int((df['Churn'] == 'No').sum()),
            "churn": int((df['Churn'] == 'Yes').sum()),
            "no_churn_percentage": round((df['Churn'] == 'No').sum() / len(df) * 100, 2),
            "churn_percentage": round((df['Churn'] == 'Yes').sum() / len(df) * 100, 2)
        },
        "columns": df.columns.tolist()
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print("API available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

