import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.title("Student Stress Level Prediction")

@st.cache_resource
def load_model():
    """
    Loads the model from the 'models' directory using a robust, absolute path.
    This prevents FileNotFoundError by not relying on the current working directory.
    """
    # Get the absolute path of the directory where this script is located
    app_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the path to the model file (../models/stress_level_model.pkl)
    model_path = os.path.join(app_dir, '..', 'model', 'stress_level_model.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at the expected path: {model_path}")
        st.error("Please run the `model.ipynb` notebook to generate the model file.")
        return None
    
    return joblib.load(model_path)

model = load_model()

# Define the exact feature order the model was trained on
FEATURE_ORDER = [
    'anxiety_level', 'self_esteem', 'mental_health_history', 'depression',
    'headache', 'blood_pressure', 'sleep_quality', 'breathing_problem',
    'noise_level', 'living_conditions', 'safety', 'basic_needs',
    'academic_performance', 'study_load', 'teacher_student_relationship',
    'future_career_concerns', 'social_support', 'peer_pressure',
    'extracurricular_activities', 'bullying'
]

# Define stress level mapping for clear output
STRESS_LEVEL_MAP = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}

# Create input form
st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Patient Name")

# Use a dictionary to store inputs, making them easy to manage
inputs = {}

col1, col2 = st.columns(2)
with col1:
    inputs['anxiety_level'] = st.slider("Anxiety Level", 0, 21, 10)
    inputs['self_esteem'] = st.slider("Self Esteem", 0, 30, 15)
    inputs['mental_health_history'] = st.selectbox("Mental Health History", [0, 1])
    inputs['depression'] = st.slider("Depression Level", 0, 27, 14)
    inputs['headache'] = st.slider("Headache Frequency", 0, 5, 2)
    inputs['blood_pressure'] = st.slider("Blood Pressure Level", 1, 3, 2)
    inputs['sleep_quality'] = st.slider("Sleep Quality", 0, 5, 2)
    inputs['breathing_problem'] = st.slider("Breathing Problem", 0, 5, 2)
    inputs['noise_level'] = st.slider("Noise Level", 0, 5, 2)
    inputs['living_conditions'] = st.slider("Living Conditions", 0, 5, 2)

with col2:
    inputs['safety'] = st.slider("Safety Level", 0, 5, 2)
    inputs['basic_needs'] = st.slider("Basic Needs Met", 0, 5, 2)
    inputs['academic_performance'] = st.slider("Academic Performance", 0, 5, 2)
    inputs['study_load'] = st.slider("Study Load", 0, 5, 2)
    inputs['teacher_student_relationship'] = st.slider("Teacher-Student Relationship", 0, 5, 2)
    inputs['future_career_concerns'] = st.slider("Future Career Concerns", 0, 5, 2)
    inputs['social_support'] = st.slider("Social Support", 0, 3, 2)
    inputs['peer_pressure'] = st.slider("Peer Pressure", 0, 5, 2)
    inputs['extracurricular_activities'] = st.slider("Extracurricular Activities", 0, 5, 2)
    inputs['bullying'] = st.slider("Bullying", 0, 5, 2)

# Create prediction button
if st.button("Predict Stress Level"):
    if model is not None:
        # Create a pandas DataFrame with the correct feature names and order
        input_df = pd.DataFrame([inputs], columns=FEATURE_ORDER)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display results
        st.header(f"Results for {patient_name or 'the Patient'}")
        predicted_stress_level = STRESS_LEVEL_MAP.get(prediction, "Unknown")
        st.subheader(f"Predicted Stress Level: {predicted_stress_level}")
        
        # Create probability chart
        prob_df = pd.DataFrame({'Stress Level': list(STRESS_LEVEL_MAP.values()), 'Probability': prediction_proba})
        fig = px.bar(prob_df, x='Stress Level', y='Probability', title='Prediction Probabilities')
        st.plotly_chart(fig, use_container_width=True)