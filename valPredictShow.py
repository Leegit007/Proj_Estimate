import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os
from datetime import date, datetime

# Define the custom objects when loading the model
custom_objects = {
    'MeanSquaredError': MeanSquaredError,
    'mse': MeanSquaredError()
}

# Load the pre-trained models
model_dir = "./model"

models = {
    'RandomForest': joblib.load(os.path.join(model_dir, "RandomForest.pkl")),
    'LightGBM': joblib.load(os.path.join(model_dir, "LightGBM.pkl")),
    'MultiTaskLasso': joblib.load(os.path.join(model_dir, "MRM.pkl")),
    'MLP': joblib.load(os.path.join(model_dir, "MLP.pkl")),
    # 'MTNN': load_model(os.path.join(model_dir, "MTNN_model.h5"), custom_objects=custom_objects)
}

# Feature and output columns definitions
input_cols = ['ECC', 'S/4HANA', 'BTP', 'RAP', 'CAP', 'DATAREPLICATION', 'BAS', 'MOBILEDEVLOPMENT', 'GENAI', 'NARROWAI']
output_cols = {
    'UI': 'User Interface',
    'BE': 'Backend',
    'CNF': 'Including Cloud Configuration',
    'FUNAI': 'Functional + AI',
    'UX': 'User Experience',
    'TRANSLATION': 'Translation',
    'TESTING': 'Testing',
    'SPRINT0': 'Sprint 0'
}

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'input_data_list' not in st.session_state:
    st.session_state.input_data_list = []
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

# Application Title
st.title("Project Estimate Prediction Application")

# Prepare features for prediction
def prepare_features(selected_features, nopack, complexity, region):
    features = {col: 1 if col in selected_features else 0 for col in input_cols}
    features['NOPACK'] = nopack
    features['COMPLEXITY'] = complexity
    features['REGION'] = region
    return features

# Predict using models
def predict(features):
    predictions = {}
    input_data = pd.DataFrame([features])
    for idx, (model_name, model) in enumerate(models.items()):
        model_key = f"Model-{idx+1}"
        try:
            pred = model.predict(input_data)
            pred = np.round(np.maximum(pred, 0))
            if pred.ndim == 2 and pred.shape[1] == len(output_cols):
                pred = pred[0]
            elif pred.ndim == 1 and len(pred) == len(output_cols):
                pass
            else:
                st.error(f"Unexpected prediction shape from model {model_key}: {pred.shape}")
                pred = np.zeros(len(output_cols))
            predictions[model_key] = pred
        except Exception as e:
            st.error(f"Error with model {model_key}: {e}")
            predictions[model_key] = np.zeros(len(output_cols))
    return predictions

# Save feedback to file
def save_feedback(feedback_text):
    with open("user_feedback.txt", "a") as f:
        feedback = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": feedback_text
        }
        f.write(str(feedback) + "\n")

# Input layout within a form
with st.form("Input Form"):
    st.subheader("Input Details")
    cols = st.columns(2)
    
    with cols[0]:
        project_id = st.text_input("Project ID")
        description = st.text_area("Description", height=100)
        methodology = st.selectbox("Methodology", ["Scrum", "Kanban", "Waterfall", "Agile"], index=0)
        industry = st.selectbox("Industry", ["Cross", "Technology", "Finance", "Healthcare", "Manufacturing"], index=0)
        start_date = st.date_input("Start Date", date.today())
    
    with cols[1]:
        selected_features = st.multiselect("Select Technologies (mandatory)", input_cols)
        if not selected_features:
            st.error("Please select at least one technology.")
    
        nopack = st.slider("Number of Applications", min_value=1, max_value=10, value=1)
        complexity = st.selectbox("Complexity", ["Low", "Medium", "High", "Very High"], index=0)
        complexity_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
        region = st.selectbox("Region", ["APJ", "EMEA", "Americas"], index=0)
        region_map = {"APJ": 1, "EMEA": 10, "Americas": 100}
    
    # Add a submit button to the form
    submitted = st.form_submit_button("Predict")

    if submitted:
        if not selected_features:
            st.error("Please select at least one technology before submitting.")
        else:
            features = prepare_features(selected_features, nopack, complexity_map[complexity], region_map[region])
            predictions = predict(features)
        
            # Determine the next prediction number
            prediction_number = len(st.session_state.input_data_list) // len(models) + 1
        
            # Store input data and predictions
            for model_name, pred in predictions.items():
                input_data = {
                    "Prediction Number": f"Prediction {prediction_number}",
                    "Project ID": project_id,
                    "Description": description,
                    "Methodology": methodology,
                    "Industry": industry,
                    "Start Date": start_date,
                    "Technologies": ", ".join(selected_features),
                    "Number of Applications": nopack,
                    "Complexity": complexity,
                    "Region": region,
                    "Prediction Set": model_name
                }
                for key, value in output_cols.items():
                    input_data[value] = pred[list(output_cols.keys()).index(key)]
                input_data["Total"] = np.sum(pred)
                st.session_state.input_data_list.append(input_data)
    
            st.session_state.feedback_submitted = False  # Reset feedback submission status
            st.success("Prediction made successfully!")

# Display combined results with full screen width
if st.session_state.input_data_list:
    st.subheader("Combined Input Data and Predictions")
    df = pd.DataFrame(st.session_state.input_data_list)
    st.dataframe(df, use_container_width=True)
    
    # Ask for feedback
    st.subheader("Feedback")
    if not st.session_state.feedback_submitted:
        feedback_text = st.text_area("Please provide your feedback here and hit submit:")
        if st.button("Submit Feedback"):
            save_feedback(feedback_text)
            st.session_state.feedback_submitted = True
            st.success("Thank you for your feedback!")
    else:
        st.write("# **Thank you for your feedback!**")