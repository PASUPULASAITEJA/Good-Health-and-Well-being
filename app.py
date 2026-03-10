import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Chronic Disease Prediction",
    page_icon="⚕️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f8fbfa; }
    .stButton>button {
        background-color: #106b6b; color: white; font-weight: bold;
        border-radius: 8px; padding: 12px 24px; width: 100%;
        border: none; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0c4d4d; transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-card {
        padding: 24px; border-radius: 12px; text-align: center;
        margin-top: 24px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .high-risk { background-color: #fff0f0; color: #d32f2f; border-left: 6px solid #d32f2f; }
    .low-risk { background-color: #f1f8e9; color: #2e7d32; border-left: 6px solid #2e7d32; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(disease):
    paths = {
        "Diabetes": "models/diabetes_model.pkl",
        "Heart Disease": "models/heart_model.pkl",
        "Breast Cancer": "models/cancer_model.pkl"
    }
    path = paths.get(disease)
    if path and os.path.exists(path):
        data = joblib.load(path)
        return data['model'], data['scaler'], data.get('features', [])
    return None, None, None

def show_result(prediction, probability, disease):
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card high-risk">
            <h2>⚠️ High Risk of {disease}</h2>
            <h4>Probability: {probability:.1f}%</h4>
            <div style="margin-top: 20px; text-align: left; background-color: white; padding: 15px; border-radius: 8px;">
                <h4 style="margin-top: 0; color: #333;">Recommended Actions:</h4>
                <ul style="color: #555; line-height: 1.6;">
                    <li><strong>Consult a Healthcare Professional immediately.</strong> Do not self-diagnose.</li>
                    <li>Follow specific medical guidance for further diagnostic testing.</li>
                    <li>Focus on a healthy diet, exercise, and stress management while awaiting consultation.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card low-risk">
            <h2>✅ Low Risk of {disease}</h2>
            <h4>Probability: {probability:.1f}%</h4>
            <div style="margin-top: 20px; text-align: left; background-color: white; padding: 15px; border-radius: 8px;">
                <h4 style="margin-top: 0; color: #333;">Health Maintenance:</h4>
                <ul style="color: #555; line-height: 1.6;">
                    <li>Maintain healthy lifestyle choices.</li>
                    <li>Continue regular medical checkups.</li>
                    <li>Engage in regular physical fitness routines.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


def diabetes_ui():
    st.title("🩸 Diabetes Prediction")
    st.markdown("Enter patient data to evaluate diabetes risk.")
    model, scaler, feature_names = load_model("Diabetes")
    if not model:
        st.warning("Model not found. Please train first.")
        return

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        age = st.number_input("Age", 1, 120, 33)
        bmi = st.number_input("BMI", 10.0, 80.0, 25.0, 0.1)
        blood_pressure = st.number_input("Blood Pressure", 20, 150, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    with col2:
        glucose = st.number_input("Glucose Level", 40, 300, 120)
        insulin = st.number_input("Insulin Level", 0, 1000, 79)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        db_pedigree = st.number_input("Diabetes Pedigree Function", 0.05, 3.0, 0.5, 0.01)

    if st.button("Predict Diabetes Risk"):
        input_dict = {
            'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
            'DiabetesPedigreeFunction': db_pedigree, 'Age': age
        }
        df = pd.DataFrame([input_dict])[feature_names]
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1] * 100
        show_result(pred, prob, "Diabetes")

def heart_disease_ui():
    st.title("🫀 Heart Disease Prediction")
    st.markdown("Enter patient data to evaluate heart disease risk.")
    model, scaler, feature_names = load_model("Heart Disease")
    if not model:
        st.warning("Model not found. Please train first.")
        return

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl ?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina?", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
        oldpeak = st.number_input("ST Depression (OldPeak)", 0.0, 10.0, 1.0, 0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
        ca = st.number_input("Number of Major Vessels Colored (0-4)", 0, 4, 0)
        thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])

    if st.button("Predict Heart Disease Risk"):
        input_dict = {
            'Age': age, 'Sex': sex, 'Chest Pain Type': cp, 'Resting Blood Pressure': trestbps,
            'Cholesterol': chol, 'Fasting Blood Sugar': fbs, 'Resting ECG': restecg,
            'Max Heart Rate': thalach, 'Exercise Induced Angina': exang, 
            'ST Depression': oldpeak, 'Slope of Peak Exercise ST Segment': slope, 
            'Major Vessels Colored by Flourosopy': ca, 'Thalassemia': thal
        }
        df = pd.DataFrame([input_dict])[feature_names]
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1] * 100
        show_result(pred, prob, "Heart Disease")


def cancer_ui():
    st.title("🎗️ Breast Cancer Prediction")
    st.markdown("Enter mean cellular feature measurements to predict cancer risk (Malignant vs Benign).")
    model, scaler, feature_names = load_model("Breast Cancer")
    if not model:
        st.warning("Model not found. Please train first.")
        return

    # To keep it user-friendly, we provide sliders for the top mean features, 
    # and default values for the rest.
    st.markdown("#### Primary Predictors (Mean values)")
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        mean_radius = st.number_input("Mean Radius", 5.0, 30.0, 14.0)
        mean_texture = st.number_input("Mean Texture", 5.0, 40.0, 19.0)
        mean_perimeter = st.number_input("Mean Perimeter", 40.0, 200.0, 91.0)
        mean_area = st.number_input("Mean Area", 100.0, 2500.0, 650.0)
        mean_smoothness = st.number_input("Mean Smoothness", 0.05, 0.25, 0.1)
    with col2:
        mean_compactness = st.number_input("Mean Compactness", 0.01, 0.35, 0.1)
        mean_concavity = st.number_input("Mean Concavity", 0.0, 0.45, 0.08)
        mean_concave_pts = st.number_input("Mean Concave Points", 0.0, 0.2, 0.04)
        mean_symmetry = st.number_input("Mean Symmetry", 0.1, 0.4, 0.18)
        mean_fractal = st.number_input("Mean Fractal Dimension", 0.04, 0.15, 0.06)

    if st.button("Predict Breast Cancer Risk (Malignancy)"):
        input_dict = {
            'mean radius': mean_radius, 'mean texture': mean_texture, 'mean perimeter': mean_perimeter,
            'mean area': mean_area, 'mean smoothness': mean_smoothness, 'mean compactness': mean_compactness,
            'mean concavity': mean_concavity, 'mean concave points': mean_concave_pts, 'mean symmetry': mean_symmetry,
            'mean fractal dimension': mean_fractal
        }
        
        # For the remaining 20 features, fill with mean logic or defaults based on breast cancer dataset characteristics
        # Standard error and worst features are the other 20. Fill with zeros or basic defaults for demonstration.
        for f in feature_names:
            if f not in input_dict:
                input_dict[f] = 0.0
                
        df = pd.DataFrame([input_dict])[feature_names]
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1] * 100
        show_result(pred, prob, "Breast Cancer")

def main():
    st.sidebar.title("⚕️ AI Diagnostic System")
    st.sidebar.markdown("Select an disease module below:")
    
    app_mode = st.sidebar.selectbox("Disease Predictor", ["Diabetes", "Heart Disease", "Breast Cancer"])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This system is built to utilize Machine Learning to predict likelihood of chronic diseases. For educational purposes only.")

    if app_mode == "Diabetes":
        diabetes_ui()
    elif app_mode == "Heart Disease":
        heart_disease_ui()
    elif app_mode == "Breast Cancer":
        cancer_ui()

if __name__ == "__main__":
    main()
