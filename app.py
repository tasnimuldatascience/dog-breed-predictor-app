# Library imports
import numpy as np
import streamlit as st
import cv2
from tensorflow.keras.models import load_model

# Load the Model
model = load_model('dog_breed.h5')

# Class Names
CLASS_NAMES = ['Scottish Deerhound', 'Maltese Dog', 'Bernese Mountain Dog']

# Streamlit App Title
st.title("Dog Breed Prediction")
st.markdown("Upload an image of the dog")

# Upload Image
dog_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
submit = st.button('Predict')

# On button click
if submit:
    if dog_image is not None:
        # Convert file to OpenCV format
        file_bytes = np.asarray(bytearray(dog_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if opencv_image is None:
            st.error("Error loading the image. Please upload a valid image file.")
        else:
            # Display the uploaded image
            st.image(opencv_image, channels="BGR", caption="Uploaded Image")

            # Resize the image to match model input size
            opencv_image = cv2.resize(opencv_image, (224, 224))

            # Convert to correct shape for model
            opencv_image = np.reshape(opencv_image, (1, 224, 224, 3)) / 255.0  # Normalize pixel values

            # Make Prediction
            Y_pred = model.predict(opencv_image)

            # Display Prediction
            predicted_breed = CLASS_NAMES[np.argmax(Y_pred)]
            st.title(f"The Dog Breed is: {predicted_breed}")
    else:
        st.error("Please upload an image before clicking Predict.")
