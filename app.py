import streamlit as st
import pandas as pd
from ucla_admissions.modeling.predict import predict_admission
from ucla_admissions.modeling.train import load_or_train_model
from ucla_admissions.features import preprocess_training_data, preprocess_user_input
from ucla_admissions.dataset import load_data
import logging
from ucla_admissions.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE
import os
from ucla_admissions.plots import plot_loss_curve


# Ensure logs directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

st.title("🎓 UCLA Admissions Prediction")

st.markdown("Enter your details below to predict your chance of admission:")

# User Inputs
gre_score = st.slider("GRE Score", 260, 340, 320)
toefl_score = st.slider("TOEFL Score", 0, 120, 110)
univ_rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
sop = st.slider("Statement of Purpose Strength", 1.0, 5.0, 3.5, step=0.5)
lor = st.slider("Letter of Recommendation Strength", 1.0, 5.0, 3.5, step=0.5)
cgpa = st.slider("Undergraduate CGPA", 0.0, 10.0, 8.0, step=0.1)
research = st.radio("Research Experience", options=["Yes", "No"])

# Prepare user input dataframe
user_data = pd.DataFrame(
    {
        "GRE_Score": [gre_score],
        "TOEFL_Score": [toefl_score],
        "University_Rating": [str(univ_rating)],
        "SOP": [sop],
        "LOR": [lor],
        "CGPA": [cgpa],
        "Research": ["1" if research == "Yes" else "0"],
    }
)

# Load and preprocess training data
df = load_data()
X_scaled, y_train, scaler, feature_columns = preprocess_training_data(df)

# Train or load model
model = load_or_train_model(X_scaled, y_train)

# Preprocess user input based on training metadata
X_user_scaled = preprocess_user_input(user_data, feature_columns, scaler)


# Predict with trained model
prediction, probability = predict_admission(X_user_scaled, model)


# Display result
if prediction[0] == 1:
    st.success(f"👍 High chance of admission! Probability: {probability[0] * 100:.2f}%")
else:
    st.warning(f"⚠️ Lower chance of admission. Probability: {probability[0] * 100:.2f}%")

if st.checkbox("📉 Show Loss Curve"):
    st.markdown("### Model Training Loss Curve")
    fig = plot_loss_curve(model)
    st.pyplot(fig)
