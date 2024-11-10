
import streamlit as st
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Title of the app
st.title("Penguin Species Prediction")

# Upload the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the dataframe
    st.write("Dataset Preview", df.head())

    # Separate features and target
    X = df.drop('species', axis=1)
    y = df['species']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical features (features containing 'Biscoe' or other non-numeric values)
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Create a pipeline with an encoder for categorical features
    model = Pipeline(steps=[
        ('encoder', ce.OrdinalEncoder(cols=categorical_features)),  # Encode categorical features
        ('scaler', StandardScaler()), 
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Save the model
    with open('model_penguin_66130701701.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Model inference section
    st.subheader("Make Predictions")

    # Input fields for new data
    input_data = {}
    for column in X.columns:
        input_data[column] = st.text_input(f"Enter {column}", "")

    if st.button("Predict"):
        # Convert the input data into a DataFrame for prediction
        input_df = pd.DataFrame([input_data])
        
        # Predict the species
        prediction = model.predict(input_df)

        st.write(f"The predicted species is: {prediction[0]}")
