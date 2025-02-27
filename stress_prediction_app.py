
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Set page configuration
st.set_page_config(page_title="Stress Level Prediction", layout="wide")

# Add title and description
st.title("Stress Level Prediction System")
st.write("Enter patient details to predict stress level")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('stress_level_model.pkl')

model = load_model()

# Create input form
st.sidebar.header("Patient Information")
patient_name = st.sidebar.text_input("Patient Name")

# Create input fields for all features
col1, col2 = st.columns(2)

with col1:
    anxiety_level = st.slider("Anxiety Level", 0, 20, 10)
    self_esteem = st.slider("Self Esteem", 0, 20, 10)
    mental_health_history = st.selectbox("Mental Health History", [0, 1])
    depression = st.slider("Depression Level", 0, 20, 10)
    headache = st.slider("Headache Frequency", 0, 5, 2)
    blood_pressure = st.slider("Blood Pressure Level", 0, 5, 2)
    sleep_quality = st.slider("Sleep Quality", 0, 5, 2)
    breathing_problem = st.slider("Breathing Problem", 0, 5, 2)
    noise_level = st.slider("Noise Level", 0, 5, 2)
    living_conditions = st.slider("Living Conditions", 0, 5, 2)

with col2:
    safety = st.slider("Safety Level", 0, 5, 2)
    basic_needs = st.slider("Basic Needs Met", 0, 5, 2)
    academic_performance = st.slider("Academic Performance", 0, 5, 2)
    study_load = st.slider("Study Load", 0, 5, 2)
    teacher_student_relationship = st.slider("Teacher-Student Relationship", 0, 5, 2)
    future_career_concerns = st.slider("Future Career Concerns", 0, 5, 2)
    social_support = st.slider("Social Support", 0, 5, 2)
    peer_pressure = st.slider("Peer Pressure", 0, 5, 2)
    extracurricular_activities = st.slider("Extracurricular Activities", 0, 5, 2)
    bullying = st.slider("Bullying", 0, 5, 2)

# Create prediction button
if st.button("Predict Stress Level"):
    # Create input array for prediction
    input_data = np.array([[
        anxiety_level, self_esteem, mental_health_history, depression,
        headache, blood_pressure, sleep_quality, breathing_problem,
        noise_level, living_conditions, safety, basic_needs,
        academic_performance, study_load, teacher_student_relationship,
        future_career_concerns, social_support, peer_pressure,
        extracurricular_activities, bullying
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Display results
    st.header(f"Results for {patient_name}")
    
    # Display stress level
    stress_levels = ["Low Stress", "Medium Stress", "High Stress"]
    st.subheader(f"Predicted Stress Level: {stress_levels[prediction]}")
    
    # Create probability chart
    prob_df = pd.DataFrame({
        'Stress Level': stress_levels,
        'Probability': prediction_proba
    })
    
    fig = px.bar(prob_df, x='Stress Level', y='Probability',
                 title='Prediction Probabilities',
                 color='Probability',
                 color_continuous_scale='RdYlGn_r')
    
    st.plotly_chart(fig)
    
    # Display key factors
    st.subheader("Key Contributing Factors:")
    factors_df = pd.DataFrame({
        'Factor': [
            'Anxiety Level', 'Depression Level', 'Sleep Quality',
            'Academic Performance', 'Social Support'
        ],
        'Value': [
            anxiety_level, depression, sleep_quality,
            academic_performance, social_support
        ]
    })
    
    fig2 = px.bar(factors_df, x='Factor', y='Value',
                  title='Key Metrics Overview',
                  color='Value',
                  color_continuous_scale='RdYlGn_r')
    
    st.plotly_chart(fig2)
