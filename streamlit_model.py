import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained mode
model = load_model("my_model.h5")


# Function to preprocess the input image
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to make predictions on the input image
def predict(image):
    prediction = model.predict(image)
    return prediction[0][0]


# Define the Streamlit app
def main():
    st.title("COVID-19 X-Ray Detector")

    # Display a file uploader widget
    uploaded_file = st.file_uploader(
        "Choose an X-ray image...", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        img = cv2.imdecode(
            np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED
        )
        st.image(
            img,
            channels="RGB",
            use_column_width=True,
            width=10,
            caption="Uploaded X-ray image.",
        )

        # Preprocess the image and make predictions
        img_array = preprocess_image(uploaded_file)
        prediction = predict(img_array)

        # Display the prediction
        if prediction > 0.5:
            st.write("The X-ray image is **negative** for COVID-19.")
        else:
            st.write("The X-ray image is **positive** for COVID-19.")


if __name__ == "__main__":
    main()
