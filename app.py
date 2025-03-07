import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score, precision_score, recall_score, f1_score

# Set up page layout
st.set_page_config(page_title="AutoML Trainer", page_icon="⚡", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f0f2f6; }
        h1 { text-align: center; color: #333; }
        .stButton > button { width: 100%; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)



# Function to load dataset
def load_dataset(file):
    ext = file.name.split(".")[-1]
    if ext == "csv":
        df = pd.read_csv(file)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(file)
    else:
        return "Unsupported file format!", None
    return df, file.name

# Function to preprocess data
def preprocess_data(df):
    df = df.dropna()
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, y.dtype.kind == 'f'

# Function to train model
def train_model(file, model_type):
    df, dataset_name = load_dataset(file)
    if isinstance(df, str):
        return df, None, None, None
    X, y, is_regression = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model_architecture = ""
    if is_regression:
        if model_type == "Random Forest":
            model = RandomForestRegressor()
            model_architecture = "RandomForestRegressor with default parameters"
        elif model_type == "Linear Regression":
            model = LinearRegression()
            model_architecture = "Linear Regression with ordinary least squares"
        elif model_type == "SVM":
            model = SVR()
            model_architecture = "Support Vector Regression with RBF kernel"
        else:
            return "Invalid Model Selection", None, None, None
    else:
        if model_type == "Random Forest":
            model = RandomForestClassifier()
            model_architecture = "RandomForestClassifier with default parameters"
        elif model_type == "Logistic Regression":
            model = LogisticRegression()
            model_architecture = "Logistic Regression using sigmoid function"
        elif model_type == "SVM":
            model = SVC()
            model_architecture = "Support Vector Classification with RBF kernel"
        else:
            return "Invalid Model Selection", None, None, None
    
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        progress_bar.progress(percent_complete + 1)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    if is_regression:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics_text = f"Dataset: {dataset_name}\nModel: {model_type}\nArchitecture: {model_architecture}\n\nMSE: {mse}\nR² Score: {r2}"
        fig = None
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        metrics_text = f"Dataset: {dataset_name}\nModel: {model_type}\nArchitecture: {model_architecture}\n\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {f1}\n\nClassification Report:\n{report}"
    
    # Save metrics to a file
    metrics_file = io.BytesIO()
    metrics_file.write(metrics_text.encode('utf-8'))
    metrics_file.seek(0)
    
    # Save trained model
    model_file = io.BytesIO()
    joblib.dump(model, model_file)
    model_file.seek(0)
    
    return metrics_text, fig, metrics_file, model_file

# Main UI
st.title("⚡ AutoML Trainer - Train ML Models in One Click!")


file = st.file_uploader("📂 Upload CSV/XLSX File", type=["csv", "xls", "xlsx"])
model_type = st.selectbox("🛠 Select Model", ["Random Forest", "Logistic Regression", "SVM", "Linear Regression"])

if st.button("🚀 Train Model") and file is not None:
    results, fig, metrics_file, model_file = train_model(file, model_type)
    st.success("✅ Model Training Completed!")
    st.text(results)
    if fig:
        st.pyplot(fig)
    st.download_button("📥 Download Metrics", metrics_file, file_name="metrics.txt", mime="text/plain")
    st.download_button("📥 Download Trained Model", model_file, file_name="trained_model.pkl", mime="application/octet-stream")
