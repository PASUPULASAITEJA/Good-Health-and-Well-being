import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

def train_and_evaluate(X_train, X_test, y_train, y_test, disease_name):
    # Ensure X is DataFrame for StandardScaler to learn feature names
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Best practice is to convert back to DataFrame to preserve feature names, 
    # but Streamlit inference requires keeping names dynamically if we use transform on a dataframe.
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_accuracy = 0
    best_model_name = ""
    
    print(f"\nTraining models for {disease_name}...")
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"  - {name} Accuracy: {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_model_name = name
            
    print(f"⭐ Best Model for {disease_name}: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    return best_model, scaler

# ----------------- DIABETES -----------------
def process_diabetes():
    from sklearn.datasets import fetch_openml
    print("\nFetching Diabetes dataset...")
    diabetes = fetch_openml(name='diabetes', version=1, as_frame=True, parser='auto')
    df = diabetes.frame
    
    rename_map = {
        'preg': 'Pregnancies', 'plas': 'Glucose', 'pres': 'BloodPressure',
        'skin': 'SkinThickness', 'insu': 'Insulin', 'mass': 'BMI',
        'pedi': 'DiabetesPedigreeFunction', 'age': 'Age', 'class': 'Outcome'
    }
    df = df.rename(columns=rename_map)
    df['Outcome'] = df['Outcome'].map({'tested_positive': 1, 'tested_negative': 0})
    
    cols_with_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_missing:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
        
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, "Diabetes")
    
    # Include ordered columns in a metadata dict so app.py knows the exact features
    joblib.dump({'model': best_model, 'scaler': scaler, 'features': list(X.columns)}, 'models/diabetes_model.pkl')

# ----------------- BREAST CANCER -----------------
def process_breast_cancer():
    from sklearn.datasets import load_breast_cancer
    print("\nFetching Breast Cancer dataset...")
    bc = load_breast_cancer(as_frame=True)
    df = bc.frame
    X = bc.data
    y = bc.target # 0: malignant, 1: benign (Sklearn default)
    # Let's map so 1 is "High Risk" (malignant) and 0 is "Low Risk" (benign)
    y = y.map({0: 1, 1: 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, "Breast Cancer")
    
    joblib.dump({'model': best_model, 'scaler': scaler, 'features': list(X.columns)}, 'models/cancer_model.pkl')

# ----------------- HEART DISEASE -----------------
def process_heart_disease():
    print("\nFetching Heart Disease dataset from UCI...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # UCI dataset doesn't have headers in the CSV
    cols = ['Age', 'Sex', 'Chest Pain Type', 'Resting Blood Pressure', 'Cholesterol', 
            'Fasting Blood Sugar', 'Resting ECG', 'Max Heart Rate', 'Exercise Induced Angina', 
            'ST Depression', 'Slope of Peak Exercise ST Segment', 
            'Major Vessels Colored by Flourosopy', 'Thalassemia', 'Outcome']
            
    try:
        df = pd.read_csv(url, names=cols, na_values='?')
    except Exception as e:
        print(f"Failed to download heart disease dataset: {e}")
        return

    # Handle missing values ('?') which were parsed as NaNs
    df = df.dropna()

    # The Outcome is 0 for absence, 1,2,3,4 for presence of heart disease
    df['Outcome'] = (df['Outcome'] > 0).astype(int)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model, scaler = train_and_evaluate(X_train, X_test, y_train, y_test, "Heart Disease")
    
    joblib.dump({'model': best_model, 'scaler': scaler, 'features': list(X.columns)}, 'models/heart_model.pkl')


if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    process_diabetes()
    process_breast_cancer()
    process_heart_disease()
    
    print("\nAll models trained and saved successfully in 'models/' directory.")
