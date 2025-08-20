import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
st.title("🌸 Iris Flower Classification App")
st.markdown("Enter the flower measurements to predict its species.")

# Input fields (Number Input instead of Sliders)
sepal_length = st.number_input("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.number_input("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.number_input("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.number_input("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Predict button
if st.button("Predict"):
    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(sample)
    predicted_species = iris.target_names[prediction[0]]
    
    st.success(f"🌟 Predicted Species: *{predicted_species.capitalize()}*")
