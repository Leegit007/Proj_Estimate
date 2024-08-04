import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import os
from datetime import date

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

# Define feature columns
input_cols = ['ECC', 'S/4HANA', 'BTP', 'RAP', 'CAP', 'DATAREPLICATION', 'BAS', 'MOBILEDEVLOPMENT', 'GENAI', 'NARROWAI']
output_cols = ['UI', 'BE', 'CNF', 'FUNAI', 'UX', 'TRANSLATION', 'TESTING', 'SPRINT0']

# Define the Streamlit application
st.title("Project Estimate Prediction Application")

# Layout with multiple columns
col1, col2 = st.columns(2)

with col1:
    project_id = st.text_input("Project ID")
    description = st.text_area("Description")
    methodology = st.selectbox("Methodology", ["Scrum", "Kanban", "Waterfall", "Agile"], index=0)
    industry = st.selectbox("Industry", ["Cross", "Technology", "Finance", "Healthcare", "Manufacturing"], index=0)
    start_date = st.date_input("Start Date", date.today())

with col2:
    selected_features = st.multiselect("Select Technologies (mandatory)", input_cols)
    if not selected_features:
        st.error("Please select at least one technology.")
        st.stop()

    nopack = st.number_input("Number of Applications", min_value=1, max_value=10, value=1)
    complexity = st.selectbox("Complexity", ["Low", "Medium", "High", "Very High"], index=0)
    complexity_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    region = st.selectbox("Region", ["APJ", "EMEA", "Americas"], index=0)
    region_map = {"APJ": 1, "EMEA": 10, "Americas": 100}

# Prepare the input features for prediction
features = {col: 1 if col in selected_features else 0 for col in input_cols}
features['NOPACK'] = nopack
features['COMPLEXITY'] = complexity_map[complexity]
features['REGION'] = region_map[region]

# Convert to DataFrame
input_data = pd.DataFrame([features])

# Predict button
if st.button("Predict"):
    predictions = {}
    
    for idx, (model_name, model) in enumerate(models.items()):
        generic_model_name = f"Model-{idx+1}"
        try:
            pred = model.predict(input_data)
            pred = np.round(np.maximum(pred, 0))
            
            # Debug: Print the shape and content of the predictions
            print(f"{generic_model_name} prediction shape: {pred.shape}")
            print(f"{generic_model_name} prediction content: {pred}")

            if pred.ndim == 2 and pred.shape[1] == len(output_cols):
                pred = pred[0]
            elif pred.ndim == 1 and len(pred) == len(output_cols):
                pass
            else:
                st.error(f"Unexpected prediction shape from model {generic_model_name}: {pred.shape}")
                pred = np.zeros(len(output_cols))

            predictions[generic_model_name] = pred

        except Exception as e:
            st.error(f"Error with model {generic_model_name}: {e}")
            predictions[generic_model_name] = np.zeros(len(output_cols))

    # Display results
    st.write("Predictions:")
    for idx, (generic_model_name, pred) in enumerate(predictions.items()):
        total = np.sum(pred)
        result_dict = {col: int(pred[i]) for i, col in enumerate(output_cols)}
        st.write(f"**{generic_model_name}**: {result_dict}, **Total**: **{total}**")
