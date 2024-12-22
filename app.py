# Import Libraries
import streamlit as st
import numpy as np
from lime import lime_image
import tensorflow as tf
from skimage.segmentation import mark_boundaries
import keras
from PIL import Image

st.title('Explainable Alzheimer Detection and Classification using LIME')
CLASS_NAMES = ['Mild_Demented', 'Moderate_Demented', 'Non_Demented', 'Very_Mild_Demented']
@st.cache_resource(show_spinner=True,)
def get_ml_model():
    model = keras.models.load_model("model.keras")
    return model


    
def predict_fn(images):
    model = get_ml_model()
    # Batch prediction
    preds = model.predict(images)
    # Return probabilities
    return tf.nn.softmax(preds).numpy()

@st.cache_data(show_spinner=True)
def get_expln(_tfimage,pilimage):
    processed_array = np.array(pilimage)
    processed_image = _tfimage
    if tf.is_tensor(processed_image):
        processed_image = processed_image.numpy()
    # Ensure image is in uint8 format
    processed_image = processed_image.astype('uint8')

    explainer = lime_image.LimeImageExplainer()
    expln = explainer.explain_instance(processed_image, 
                                       predict_fn, 
                                       top_labels=5, 
                                       hide_color=1, 
                                       num_samples=1000) 
    temp, mask = expln.get_image_and_mask(expln.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    
    boundaries = mark_boundaries(temp / 255.0,
                                 mask,
                                 color=(1, 1, 0)
                                 )
    # black_mask = processed_array < 10
    # # expanded_black_mask = np.expand_dims(black_mask, axis=-1)
    # # filtered = np.where(expanded_black_mask, boundaries, 255)
    # filtered_image = Image.fromarray(boundaries.astype(np.uint8))
    
    return expln.top_labels[0],boundaries


file = st.file_uploader(
    "Upload an image", 
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    )

if file:
    tfimage = tf.io.decode_image(file.read(), channels=3)
    pilimage = Image.open(file).convert("L")
    with st.expander("Original Image"):  
        st.image(file)
    with st.expander("Explanation"):
        label,img = get_expln(tfimage,pilimage)
        st.text(f"Predicted Label: {CLASS_NAMES[label]}")
        st.image(image=img,clamp=True)
        