import streamlit as st
import joblib
import pandas as pd

# Load the pipeline
try:
    pipeline = joblib.load('pelletizer_pipeline.joblib')
except FileNotFoundError:
    st.error("Error: pipeline file not found. Make sure 'pelletizer_pipeline.joblib' is in the same directory.")
    st.stop()

# Set up the Streamlit application
st.title('Pelletizer Output Prediction')
st.write('Enter the input features to predict the pelletizer output.')

# Create input fields for each feature
input_features = {}
input_features['tonnage_du_disque_a'] = st.number_input('Tonnage du disque a', value=180.0)
input_features['tonnage_du_disque_b'] = st.number_input('Tonnage du disque b', value=180.0)
input_features['tonnage_du_disque_c'] = st.number_input('Tonnage du disque c', value=180.0)
input_features['tonnage_du_disque_d'] = st.number_input('Tonnage du disque d', value=180.0)
input_features['tonnage_du_disque_e'] = st.number_input('Tonnage du disque e', value=180.0)
input_features['ratio_liant'] = st.number_input('Ratio liant', value=0.1)
input_features['ratio_bentonite'] = st.number_input('Ratio bentonite', value=4.3)
input_features['45um'] = st.number_input('45um', value=73.0)
input_features['humidity'] = st.number_input('Humidity', value=8.0)

# Add a button to trigger the prediction
if st.button('Predict'):
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([input_features])

    # Make a prediction
    prediction = pipeline.predict(input_df)

    # Display the prediction result
    st.success(f'Predicted Pelletizer Output: {prediction[0]:.2f}')
