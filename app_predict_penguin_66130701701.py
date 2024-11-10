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
    # User input for numerical features
    bill_length_mm = st.number_input('Bill Length (mm)', min_value=0.0, max_value=100.0, value=40.0)
    bill_depth_mm = st.number_input('Bill Depth (mm)', min_value=0.0, max_value=100.0, value=20.0)
    flipper_length_mm = st.number_input('Flipper Length (mm)', min_value=0.0, max_value=250.0, value=200.0)
    body_mass_g = st.number_input('Body Mass (g)', min_value=0, max_value=10000, value=5000)

    # User input for categorical features
    island = st.selectbox('Island', ['Torgersen', 'Biscoe', 'Dream'])
    sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'bill_length_mm': [bill_length_mm],
        'bill_depth_mm': [bill_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'island': [island],
        'sex': [sex],
        'culmen_length_mm': [None],  # Or provide a default value if needed
        'culmen_depth_mm': [None]    # Or provide a default value if needed
    })

    return input_data

# Get user input
input_df = user_input_features()

# Ensure that the columns match the model's training data
required_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'island', 'sex', 'culmen_length_mm', 'culmen_depth_mm']
input_df = input_df[required_columns]  # Ensure only the expected columns are included

# Preprocessing: Apply categorical encoding and scaling
encoder = ce.OrdinalEncoder(cols=['island', 'sex'])
input_df_encoded = encoder.fit_transform(input_df)  # Transform the data using the encoder

# Check if the encoding process is consistent
print(input_df_encoded.columns)  # Check the columns after encoding

# Scale the numerical features (based on your model's training process)
scaler = StandardScaler()
numerical_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'culmen_length_mm', 'culmen_depth_mm']
input_df_encoded[numerical_columns] = scaler.fit_transform(input_df_encoded[numerical_columns])

# Make predictions based on the preprocessed input
prediction = model.predict(input_df_encoded)

# Display the prediction result
st.write(f'Predicted species: {prediction[0]}')
