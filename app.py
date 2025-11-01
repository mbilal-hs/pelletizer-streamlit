import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model and preprocessors
try:
    regressor = joblib.load('pelletizer_model.joblib')
    sc = joblib.load('pelletizer_scaler.joblib')
    imputer = joblib.load('pelletizer_imputer.joblib')
except FileNotFoundError:
    st.error("Error: Model or preprocessor files not found. Make sure 'pelletizer_model.joblib', 'pelletizer_scaler.joblib', and 'pelletizer_imputer.joblib' are in the same directory.")
    st.stop()

# Define the column names used during training, ensuring correct order
# Assuming the order is tonnage_du_disque_a to e, then ratio_liant, ratio_bentonite, 45um, humidity
feature_column_names = [
    'tonnage_du_disque_a', 'tonnage_du_disque_b', 'tonnage_du_disque_c',
    'tonnage_du_disque_d', 'tonnage_du_disque_e', 'ratio_liant',
    'ratio_bentonite', '45um', 'humidity'
]

# Create the Streamlit application layout
st.title('Pelletizer Output Prediction')

st.write("Enter the input parameters for the pelletizer discs and other features to predict the final output.")

# Create input widgets for each feature
input_data = {}

st.subheader("Pelletizer Disc Tonnages (tons)")
col1, col2, col3 = st.columns(3)
with col1:
    input_data['tonnage_du_disque_a'] = st.number_input('Disc A Tonnage', min_value=0.0, value=180.0, step=1.0)
    input_data['tonnage_du_disque_d'] = st.number_input('Disc D Tonnage', min_value=0.0, value=180.0, step=1.0)
with col2:
    input_data['tonnage_du_disque_b'] = st.number_input('Disc B Tonnage', min_value=0.0, value=180.0, step=1.0)
    input_data['tonnage_du_disque_e'] = st.number_input('Disc E Tonnage', min_value=0.0, value=180.0, step=1.0)
with col3:
    input_data['tonnage_du_disque_c'] = st.number_input('Disc C Tonnage', min_value=0.0, value=180.0, step=1.0)

st.subheader("Other Features")
col4, col5 = st.columns(2)
with col4:
    input_data['ratio_liant'] = st.number_input('Ratio Liant', min_value=0.0, value=0.4, step=0.01)
    input_data['45um'] = st.number_input('45um', min_value=0.0, value=72.0, step=0.1)
with col5:
    input_data['ratio_bentonite'] = st.number_input('Ratio Bentonite', min_value=0.0, value=4.0, step=0.01)
    input_data['humidity'] = st.number_input('Humidity', min_value=0.0, value=8.0, step=0.1)


# Create a button to trigger prediction
if st.button('Predict Output'):
    # Collect user inputs into a DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_column_names)

    # Impute missing values (though with number_input, missing values are less likely)
    # Still good practice to include if the app were to handle other input methods
    input_imputed = imputer.transform(input_df)

    # Scale the input data
    input_scaled = sc.transform(input_imputed)

    # Make prediction
    predicted_output = regressor.predict(input_scaled)

    # Display the predicted output
    st.subheader('Predicted Pelletizer Output:')
    st.success(f'{predicted_output[0]:.2f}')
