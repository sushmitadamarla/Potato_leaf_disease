import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

# Function for model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\Sushmita\OneDrive\Desktop\useful tips\Edunet\Edunet Microsoft\trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))  # Preprocess image
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Normalize image
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Streamlit UI
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.radio("Select Page", ["Home", "Disease Recognition"])

# Display homepage
if app_mode == "Home":
    st.markdown("<h1 style='text-align: center;'>Potato Leaf Disease Detection System</h1>", unsafe_allow_html=True)
    st.image("Diseases.webp", use_column_width=True)  # Display an image

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Detect Potato Leaf Diseases")

    # Image Upload
    test_image = st.file_uploader("Upload a potato leaf image", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result_index = model_prediction(test_image)
        class_names = ['Early Blight', 'Late Blight', 'Healthy']
        st.success(f"Prediction: The leaf is classified as **{class_names[result_index]}**")
