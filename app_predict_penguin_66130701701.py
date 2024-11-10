import streamlit as st
import pandas as pd
import pickle
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

# Load the pre-trained model
with open('model_penguin_66130701701.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to take user input for prediction
def user_input_features():
    # User input for numerical features (adjusted to match the model's expected input)
    culmen_length_mm = st.number_input('Culmen Length (mm)', min_value=0.0, max_value=100.0, value=40.0)
    culmen_depth_mm = st.number_input('Culmen Depth (mm)', min_value=0.0, max_value=100.0, value=20.0)
    flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=250.0, value=200.0)
    body_mass_g = st.number_input('Body Mass (g)', min_value=0, max_value=10000, value=5000)

    # User input for categorical features
    island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
    sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

    # Create a DataFrame with the input features (use the correct features here)
    input_data = pd.DataFrame({
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'island': [island],
        'sex': [sex]
    })

    return input_data

# Get user input
input_df = user_input_features()

# Ensure that the columns match the model's training data (6 features expected)
required_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']
input_df = input_df[required_columns]  # Ensure only the expected columns are included

# Reorder columns to match the order used in the model training
# (Make sure you know the exact order from your model training process)
input_df = input_df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex']]

# Preprocessing: Apply categorical encoding (ensure encoding matches the model's expected encoding method)
encoder = ce.OrdinalEncoder(cols=['island', 'sex'])
input_df_encoded = encoder.fit_transform(input_df)  # Transform the data using the encoder

# Check if the encoding process is consistent
print(input_df_encoded.columns)  # Check the columns after encoding

# Scale the numerical features (based on your model's training process)
scaler = StandardScaler()
numerical_columns = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
input_df_encoded[numerical_columns] = scaler.fit_transform(input_df_encoded[numerical_columns])

# Make predictions based on the preprocessed input
prediction = model.predict(input_df_encoded)

# Display the prediction result
st.write(f'Predicted species: {prediction[0]}')
