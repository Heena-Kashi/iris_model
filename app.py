import streamlit as st
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸"
)

# Load trained model
model_path = os.path.join(os.path.dirname(__file__), "iris_model.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("Error: iris_model.pkl file not found.")
    st.info("Make sure iris_model.pkl is uploaded in the same GitHub folder as app.py.")
    st.stop()

# Title
st.title("🌸 Iris Flower Prediction")
st.write("Enter the flower measurements to predict the Iris species.")

# Input fields
sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Prediction
if st.button("Predict"):
    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(input_data)

    st.success(f"Predicted Iris Species: {prediction[0]}")


