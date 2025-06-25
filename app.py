import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import base64

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Set background image
def set_bg_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    file_ext = image_file.split('.')[-1]
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/{file_ext};base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .title-box {{
        background: rgba(255, 255, 255, 0.85);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        margin-bottom: 2rem;
    }}
    .title-box h1 {{
        color: #1a3c53;
        font-size: 2.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
    }}
    .title-box p {{
        color: #2563eb;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg_local("churn.avif")

# Centered title box
st.markdown("""
    <div class="title-box">
        <h1>Customer Churn Prediction</h1>
        <p>Predict customer retention likelihood using advanced machine learning</p>
    </div>
""", unsafe_allow_html=True)

# User input section
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 40)
        credit_score = st.number_input('Credit Score', 300, 850, 650)
        estimated_salary = st.number_input('Estimated Salary', 0, 200000, 50000)

    with col2:
        st.subheader("Account Details")
        balance = st.number_input('Balance', 0, 300000, 10000)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox('Is Active Member', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        tenure = st.slider('Tenure (Years)', 0, 10, 3)
        num_of_products = st.slider('Number of Products', 1, 4, 2)
        

if st.button('Predict Churn Probability', type='primary', use_container_width=True):
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geography]])
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.divider()
    st.subheader("Prediction Results")

    if prediction_proba > 0.5:
        st.error(f'🚨 High Churn Probability: {prediction_proba:.2%}')
        st.write(f"The customer is **likely to churn** with a {prediction_proba:.2%} probability.")
    else:
        st.success(f'✅ Low Churn Probability: {prediction_proba:.2%}')
        st.write(f"The customer is **unlikely to churn** with a {prediction_proba:.2%} probability.")

    st.progress(float(prediction_proba), text="Churn Risk Level")

st.divider()
st.caption("© 2025 Customer Retention Analytics | Powered by ANN Deep Learning Algorithm | Author: Suryansh Tripathi")
