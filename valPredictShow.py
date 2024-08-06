import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date, datetime
import logging
import traceback
import lightgbm as lgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting the application...")

# Load pre-trained models
model_dir = "./model"
models = {}

def load_model(model_name, file_path):
    logger.info(f"Loading model {model_name} from {file_path}")
    try:
        model = joblib.load(file_path)
        logger.info(f"{model_name} model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading {model_name} model: {str(e)}")
        logger.error(f"Error loading {model_name} model: {traceback.format_exc()}")
        return None

models['RandomForest'] = load_model('RandomForest', os.path.join(model_dir, "RandomForest.pkl"))
models['LightGBM'] = load_model('LightGBM', os.path.join(model_dir, "LightGBM.pkl"))

input_cols = ['ECC', 'S/4HANA', 'BTP', 'RAP', 'CAP', 'DATAREPLICATION', 'BAS', 'MOBILEDEVELOPMENT', 'GENAI', 'NARROWAI']
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

if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'input_data_list' not in st.session_state:
    st.session_state.input_data_list = []
if 'feedback_submitted' not in st.session_state:
    st.session_state.feedback_submitted = False

st.title("Project Estimate Prediction Application")

def prepare_features(selected_features, nopack, complexity, region):
    features = {col: 1 if col in selected_features else 0 for col in input_cols}
    features['NOPACK'] = nopack
    features['COMPLEXITY'] = complexity
    features['REGION'] = region
    return features

def predict(features):
    predictions = {}
    input_data = pd.DataFrame([features])
    logger.info(f"Input data for prediction: {input_data}")
    for idx, (model_name, model) in enumerate(models.items()):
        if model is None:
            logger.warning(f"{model_name} model could not be loaded.")
            continue
        model_key = f"Model-{idx+1}"
        try:
            pred = model.predict(input_data)
            logger.info(f"{model_name} prediction raw output: {pred}")
            pred = np.round(np.maximum(pred, 0))
            if pred.ndim == 2 and pred.shape[1] == len(output_cols):
                pred = pred[0]
            elif pred.ndim == 1 and len(pred) == len(output_cols):
                pass
            else:
                st.error(f"Unexpected prediction shape from {model_key}: {pred.shape}")
                logger.error(f"Unexpected prediction shape from {model_key}: {pred.shape}")
                pred = np.zeros(len(output_cols))
            predictions[model_key] = pred
        except Exception as e:
            st.error(f"Error predicting with {model_key}: {str(e)}")
            logger.error(f"Error predicting with {model_key}: {traceback.format_exc()}")
            predictions[model_key] = np.zeros(len(output_cols))
    return predictions

def save_feedback(feedback_text):
    with open("user_feedback.txt", "a") as f:
        feedback = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": feedback_text
        }
        f.write(str(feedback) + "\n")

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
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        if not selected_features:
            st.error("Please select at least one technology before submitting.")
        else:
            features = prepare_features(selected_features, nopack, complexity_map[complexity], region_map[region])
            predictions = predict(features)
        
            prediction_number = len(st.session_state.input_data_list) // len(models) + 1
        
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
    
            st.session_state.feedback_submitted = False
            st.success("Prediction made successfully!")

if st.session_state.input_data_list:
    st.subheader("Combined Input Data and Predictions")
    df = pd.DataFrame(st.session_state.input_data_list)
    st.dataframe(df, use_container_width=True)
    
    st.subheader("Feedback")
    if not st.session_state.feedback_submitted:
        feedback_text = st.text_area("Please provide your feedback here and hit submit:")
        if st.button("Submit Feedback"):
            save_feedback(feedback_text)
            st.session_state.feedback_submitted = True
            st.success("Thank you for your feedback!")
    else:
        st.write("# **Thank you for your feedback!**")
