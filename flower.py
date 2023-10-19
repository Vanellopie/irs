import streamlit as st
from joblib import load
from PIL import Image

model = load('data/iris_model.joblib')
flower_labels = ["Iris Setosa", "Iris Versicolour", "Iris Virginica"]

st.title("Iris Flower Prediction")
st.write("Enter the flower attributes in centimeters to predict the flower type:")

sepal_length = st.number_input("Sepal Length (cm):", min_value=0.0, max_value=15.0)
sepal_width = st.number_input("Sepal Width (cm):", min_value=0.0, max_value=15.0)
petal_length = st.number_input("Petal Length (cm):", min_value=0.0, max_value=15.0)
petal_width = st.number_input("Petal Width (cm):", min_value=0.0, max_value=15.0)

if st.button("Predict"):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    
    prediction = model.predict(input_data)
    predicted_class = flower_labels[prediction[0]]
    
    st.subheader("Prediction Result:")
    st.write(f"The predicted flower type is: {predicted_class}")
    
    if prediction[0] == 0:
        st.image(Image.open("images/iris_setosa.png"), caption="Iris Setosa", use_column_width=True)
    elif prediction[0] == 1:
        st.image(Image.open("images/iris_versicolour.png"), caption="Iris Versicolour", use_column_width=True)
    elif prediction[0] == 2:
        st.image(Image.open("images/iris_virginica.png"), caption="Iris Virginica", use_column_width=True)