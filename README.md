# Auto-ML-App

## Overview
This project provides an **Automatic Machine Learning (AutoML) application** built using **Streamlit**. Users can upload datasets in CSV/XLSX format, select a model, and train it automatically with preprocessing and evaluation metrics. The trained model and its performance metrics can be downloaded for further use.

## Features
- Supports multiple models: 
  - **Random Forest** (Classification & Regression)
  - **Logistic Regression** (Classification)
  - **Support Vector Machine (SVM)** (Classification & Regression)
  - **Linear Regression** (Regression)
- **Automatic Preprocessing:**
  - Handles missing values
  - Encodes categorical variables
  - Scales numerical features
- **Performance Metrics:**
  - **Classification:** Accuracy, Precision, Recall, F1-score, Confusion Matrix
  - **Regression:** Mean Squared Error (MSE), R² Score
- **Visualizations:**
  - Confusion matrix for classification
- **Downloadable Outputs:**
  - Trained model (`.pkl` file)
  - Model performance report (`metrics.txt`)

## Installation
### Prerequisites
Make sure you have Python installed (>=3.7). You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage
Run the application with:

```bash
streamlit run app.py
```

### Steps to Use:
1. Upload a dataset (CSV/XLSX format).
2. Select a model type (Random Forest, Logistic Regression, SVM, or Linear Regression).
3. Click "Train Model" to start training.
4. View the metrics and confusion matrix (for classification models).
5. Download the trained model and performance report.

## Example Dataset
For testing, you can use datasets such as:
- **Iris Dataset** (for classification)
- **Boston Housing Dataset** (for regression)

## Technologies Used
- **Streamlit** (UI framework)
- **Scikit-Learn** (Machine learning models)
- **Pandas & NumPy** (Data handling)
- **Seaborn & Matplotlib** (Visualization)
- **Joblib** (Model serialization)

## License
This project is open-source and licensed under the **MIT License**.

## Contributions
Feel free to contribute by submitting issues or pull requests!

## Author
Developed by **Tharun Bala**.
