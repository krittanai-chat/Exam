import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from sklearn.pipeline import Pipeline

# Load the pre-trained model
with open('model_penguin_66130701701.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to make predictions
def predict_penguin_species(culmen_length, culmen_depth, flipper_length, body_mass, island, sex):
    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'island': [island],
        'sex': [sex]
    })
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app layout
st.title("Penguin Species Prediction")
st.write("Enter the details of the penguin to predict its species.")

# Input fields for user
culmen_length = st.number_input("Culmen Length (mm)", min_value=0.0, max_value=100.0, value=39.1)
culmen_depth = st.number_input("Culmen Depth (mm)", min_value=0.0, max_value=100.0, value=18.7)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0, max_value=300, value=181)
body_mass = st.number_input("Body Mass (g)", min_value=0, max_value=10000, value=3750)
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'])
sex = st.selectbox("Sex", ['MALE', 'FEMALE'])

# When the user clicks the 'Predict' button
if st.button("Predict Species"):
    species = predict_penguin_species(culmen_length, culmen_depth, flipper_length, body_mass, island, sex)
    st.write(f"The predicted species is: {species}")
