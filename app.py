"""
app.py - Plant Disease Image Classifier (Streamlit App)
-----------------------------------------------------------
A simple Streamlit app that lets you upload a plant leaf image and
get a predicted disease class using the model trained in train.py.

Run with: streamlit run app.py

Make sure you have already run "python train.py" first, so that
models/plant_disease_model.keras and models/class_names.json exist.
"""

import os
import json

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
MODEL_PATH = "models/plant_disease_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMAGE_SIZE = 224  # must match the size used in train.py

st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌿 Plant Disease Image Classifier")
st.write(
    "Upload a photo of a plant leaf, and this app will predict which "
    "class (healthy or a specific disease) it most likely belongs to, "
    "using a MobileNetV2 model trained with transfer learning."
)

# ------------------------------------------------------------
# Load the trained model and class names
# ------------------------------------------------------------
# @st.cache_resource keeps the model loaded in memory instead of
# reloading it every time the user interacts with the app.
@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_NAMES_PATH):
        return None, None

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        # Keys are saved as strings in JSON, so convert them back to int
        raw_class_names = json.load(f)
        class_names = {int(k): v for k, v in raw_class_names.items()}

    return model, class_names

model, class_names = load_model_and_classes()

if model is None:
    st.error(
        "No trained model was found. Please run `python train.py` "
        "first to train the model and create the required files in "
        "the `models/` folder."
    )
    st.stop()

# ------------------------------------------------------------
# Image upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image the same way it was done during training:
    # resize to 224x224 and apply MobileNetV2's preprocessing.
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array.astype("float32"))

    # Predict
    if st.button("Predict"):
        predictions = model.predict(image_array)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = class_names[predicted_index]
        confidence = float(predictions[predicted_index]) * 100

        st.subheader("Prediction Result")
        st.metric("Predicted Class", predicted_class)
        st.metric("Confidence", f"{confidence:.2f}%")

        # Show the top 3 most likely classes as a simple table
        st.subheader("Top 3 Predictions")
        top3_indices = np.argsort(predictions)[::-1][:3]
        for idx in top3_indices:
            st.write(f"- {class_names[idx]}: {predictions[idx] * 100:.2f}%")
else:
    st.info("Please upload a leaf image to get a prediction.")

st.caption(
    "This is a student learning project. Predictions are based only "
    "on the training data used and may not always be accurate."
)
