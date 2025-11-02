import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")
st.title("👁️ Diabetic Retinopathy Classification")
st.write("Upload a retinal scan image, and the model will predict its classification.")

# --- 2. LOAD YOUR TRAINED MODEL ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    """Loads the saved .keras model."""
    try:
        model = tf.keras.models.load_model("model.keras")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

if model is None:
    st.stop()

# --- 3. DEFINE THE PREDICTION FUNCTION ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    
    *** IMPORTANT ***
    You MUST adjust the (height, width) to match the input_shape
    your model was trained on. (e.g., (224, 224))
    """
    try:
        # Open the image using PIL
        img = Image.open(image)
        
        # Resize to the target size (e.g., 224x224)
        # *** REPLACE (224, 224) WITH YOUR MODEL'S INPUT SIZE ***
        target_size = (224, 224) 
        img = img.resize(target_size)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # --- IMPORTANT: PREPROCESSING ---
        # Ensure your preprocessing here EXACTLY matches
        # what you did during training.
        #
        # Example 1: If you just scaled by 1/255.0
        img_array = img_array / 255.0
        
        # Example 2: If you used a tf.keras.applications preprocess_input
        # img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Add a batch dimension (e.g., (1, 224, 224, 3))
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Define your class names
# *** REPLACE WITH YOUR ACTUAL CLASS LABELS ***
CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

# --- 4. CREATE THE STREAMLIT INTERFACE ---
uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(uploaded_file)
    
    if preprocessed_img is not None:
        # Make a prediction
        try:
            prediction = model.predict(preprocessed_img)
            
            # Get the top prediction
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100

            # Display the result
            with col2:
                st.subheader("Prediction Result")
                st.success(f"**Diagnosis:** {predicted_class_name}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Optional: Display raw probabilities
                st.write("Full Probabilities:")
                st.dataframe({name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, prediction[0])},
                             column_config={"value": "Probability"})

        except Exception as e:
            st.error(f"Error during prediction: {e}")
