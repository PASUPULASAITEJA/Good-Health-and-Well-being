# AI-Based Chronic Disease Prediction System ⚕️

An interactive Machine Learning web application built with Streamlit that analyzes patient health data to accurately predict the risk of critical chronic diseases.

**Currently Supported Models:**
*   🩸 **Diabetes Prediction** (Pima Indians Diabetes Database)
*   🫀 **Heart Disease Prediction** (UCI Heart Disease Database)
*   🎗️ **Breast Cancer Prediction** (Scikit-Learn Wisconsin Breast Cancer Dataset)

---

## 📸 Overview
Many people develop chronic diseases such as diabetes, heart disease, or cancer without early detection. Early diagnosis is often expensive and requires physical hospital visits. 

This ML system is designed to provide rapid, accessible predictions of disease likelihood based on accessible clinical metrics alongside personalized health recommendations for better patient outcomes.

### Features
- **Multi-Disease Support**: A seamless sidebar interface allowing medical practitioners or users to switch between the diagnostic modules.
- **Robust Machine Learning**: Implements robust algorithms including Random Forest, Extra Trees, Logistic Regression, and Decision Trees depending on the dataset characteristics.
- **Smart Data Preprocessing**: Automatic handling of missing values (e.g., zeros for BMI/Glucose) and Feature Standard Scaling (`StandardScaler`).
- **Interactive UI**: A modern, colorful interface that offers immediate probability percentages and categorized medical advice.

---

## 🚀 Getting Started

Follow these steps to run the application locally on your machine.

### Prerequisites
Make sure you have **Python 3.8+** installed on your system. 

### 1. Clone or Download the Repository
Navigate to your desired folder in the terminal (Command Prompt/PowerShell):
```bash
cd path/to/your/directory
```

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a Python virtual environment to keep dependencies isolated:

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```
**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Packages
Install all necessary data science and UI libraries (Streamlit, Pandas, Scikit-learn, etc.):
```bash
pip install -r requirements.txt
```

### 4. Train the ML Models
Before starting the web app, you must fetch the datasets and train the predictive models. Run the training pipeline script:
```bash
python model_pipeline.py
```
*Note: This script will download datasets from OpenML and UCI, train the classifiers, and save the best-performing models (.pkl files) into a new `models/` directory.*

### 5. Start the Web Application
Launch the interactive Streamlit user interface:
```bash
streamlit run app.py
```
The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🧠 Technologies Used
- **Language:** Python
- **Web Framework:** Streamlit
- **Machine Learning:** Scikit-Learn
- **Data Manipulation:** Pandas, NumPy
- **Model Serialization:** Joblib

---

## ⚙️ How It Works (Pipeline)

1. **Data Collection:** `model_pipeline.py` fetches the raw clinical data (e.g., Age, Glucose Level, BMI, resting ECG, cellular area).
2. **Preprocessing:** Data is cleaned (missing values handled via median substitution) and normalized utilizing a standard scaler. Data is split into 80% training and 20% testing sets.
3. **Training:** An array of classifiers are trained concurrently. The script identifies the algorithm with the highest accuracy score on the test set.
4. **Serialization:** The winning model and the specific configuration of the feature scaler are dumped as binary `.pkl` variables using `joblib`.
5. **Inference (App):** When a user enters data via the `app.py` sliders and inputs, it dynamically reconstructs the scaler and passes the inputs to the trained model for classification.

---

## 📝 Disclaimer
*This system utilizes predictive Machine Learning for educational and demonstrative purposes only. It is not a substitute for professional medical advice, official diagnosis, or treatment. Always consult a certified healthcare professional with any concerns about your physical health.*
