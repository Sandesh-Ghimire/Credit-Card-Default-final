# Credit Card Default Risk Prediction Web App

A machine learning-based web application that predicts credit card default risk using customer financial data. This project uses advanced models like XGBoost, Logistic Regression, and Random Forest with SHAP and LIME explainability features.

## 📋 Features

- **Credit Risk Prediction**: Predicts whether a customer will default on their credit card payment
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Multiple ML Models**: XGBoost, Logistic Regression, and Random Forest
- **Model Explainability**: 
  - SHAP (SHapley Additive exPlanations) for model interpretability
  - LIME (Local Interpretable Model-agnostic Explanations) for local predictions
- **Real-time Predictions**: Input customer details and get instant risk assessment
- **Data Visualization**: Comprehensive EDA and model performance metrics

## 🎯 Project Overview

This application analyzes customer credit profiles including:
- Credit limit
- Demographics (age, gender, education, marital status)
- Repayment status history (6 months)
- Bill amounts and payment history
- Derived features (total bill, payment ratio, average delay)

## 📁 Project Structure

```
Credit-Card-Default-final/
├── creditapp.py              # Main Streamlit web application
├── optimizedcode.ipynb       # Jupyter notebook with model training & analysis
├── model.pkl                 # Pre-trained ML model
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Credit-Card-Default-final
```

### Step 2: Create a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Run the Web Application

```bash
streamlit run creditapp.py
```

The application will open in your browser at `http://localhost:8501`

**To use the web app:**
1. Enter or adjust customer details in the sidebar
2. Input credit limit, demographics, and payment history
3. Click "Predict" to get the default risk assessment
4. View predictions and model explanations

### View the Analysis Notebook

```bash
jupyter notebook optimizedcode.ipynb
```

This notebook contains:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and evaluation
- Model comparison and hyperparameter tuning
- SHAP and LIME explanations

## 📊 Model Information

The project uses multiple machine learning models:

- **XGBoost**: Gradient boosting model with high predictive power
- **Logistic Regression**: Linear baseline model for interpretability
- **Random Forest**: Ensemble method for robust predictions

**Model Performance Metrics:**
- Accuracy
- ROC-AUC Score
- F1-Score
- Precision & Recall
- Confusion Matrix

## 📝 Input Features

### Customer Demographics
- **Credit Limit**: Total credit limit (1,000 - 1,000,000)
- **Gender**: Male or Female
- **Education Level**: Graduate, University, High School, Other
- **Marital Status**: Married, Single, Other
- **Age**: 18-80 years

### Payment History
- **Repayment Status**: Last 6 months (-2 to 8 scale)
  - -2: No consumption
  - -1: Paid duly
  - 0: Paid on time
  - 1-8: Months of delay
- **Bill Amounts**: Last 6 months
- **Payment Amounts**: Last 6 months

### Derived Features
- Total Bill Amount
- Total Payment
- Payment Ratio
- Average Delay

## 🔍 Model Explainability

### SHAP Analysis
- Global feature importance
- Individual prediction explanations
- Force plots and dependency plots

### LIME Analysis
- Local model explanations
- Feature contribution for specific predictions

## 📋 Dependencies

Key packages used:
- **streamlit**: Web application framework
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **xgboost**: Gradient boosting
- **shap**: Model explainability
- **lime**: Local interpretability
- **matplotlib**: Visualization
- **seaborn**: Statistical visualization
- **joblib**: Model serialization

See `requirements.txt` for complete list with versions.

## 📚 Data Source

The project uses the UCI Credit Card Default dataset containing:
- 30,000 customer records
- 24 features (demographics, credit history, payment data)
- Target: Default payment in the next month

## ⚠️ Important Notes

Before running the application, ensure you have:
1. ✅ `model.pkl` - Pre-trained model file
2. ✅ Dataset file (referenced in the notebook)
3. ✅ All dependencies installed via `requirements.txt`

## 🛠️ Troubleshooting

**Issue**: Port 8501 already in use
```bash
streamlit run creditapp.py --server.port 8502
```

**Issue**: Module not found errors
```bash
pip install --upgrade -r requirements.txt
```

**Issue**: Virtual environment not activating
- Ensure you're in the project directory
- Check Python path: `python --version`

## 📧 Support

For issues or questions, please open an issue on GitHub.

## 📄 License

This project is part of an MSC IT Data Analytics dissertation.

---

**Last Updated**: May 2026

