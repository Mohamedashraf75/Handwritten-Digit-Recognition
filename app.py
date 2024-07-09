
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# page name
st.set_page_config('Digit Recognition', page_icon= '🔢')

# example of the title, markdown, etc
st.title('Handwritten Digit Recognition 🔢')
st.caption('by Azka Redhia')

st.markdown(r'''This simple application is designed to recognize a number from 0-9 from a PNG file with a resolution of 28x28 pixels. 
            While it may not achieve 100% accuracy, but its performance is consistently high.''')
st.subheader('Have fun giving it a try!!! 😊')
# Load your model
model = tf.keras.models.load_model('mnist_model.h5')

# Example of Streamlit file uploader and processing
uploaded_image = st.file_uploader('Upload an image', type=['png', 'jpg'])

if uploaded_image is not None:
    # Open and resize the image
    img = Image.open(uploaded_image)
    img_resized = img.resize((28, 28))  # Resize to 28x28 pixels

    # Convert image to grayscale and normalize
    img_array = np.array(img_resized.convert('L'))
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape to (1, 28, 28, 1) for model input
    img_array = img_array.astype('float32') / 255.0  # Normalize pixel values

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    # Display result
    st.image(img_resized, caption='Uploaded Image')
    st.write(f'Predicted Digit: {predicted_class}')
