import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# --- IMPORTANT ASSUMPTIONS ---
# 1. We assume your model has 5 output classes for DR
NUM_CLASSES = 5
# 2. We assume these are your class names in order
CLASS_NAMES = ["0 - No DR", "1 - Mild", "2 - Moderate", "3 - Severe", "4 - Proliferative"]

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """
    Loads the ResNet18 model architecture, modifies the final layer,
    and loads the saved weights from .pth file.
    """
    # Load the pre-trained ResNet18 model
    model = models.resnet18()
    
    # Get the number of input features for the final layer
    num_ftrs = model.fc.in_features
    
    # *** IMPORTANT ***
    # Replace the final layer with a new one matching your model's output
    # (e.g., 5 classes for DR)
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # Load your saved model weights
    # We use map_location='cpu' to ensure it runs on Streamlit's CPU-only servers
    try:
        model.load_state_dict(torch.load("resnet18_diabetic_retinopathy.pth", map_location=torch.device('cpu')))
    except FileNotFoundError:
        st.error("Model file 'resnet18_diabetic_retinopathy.pth' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model state_dict: {e}")
        st.info("Ensure the model architecture (ResNet18) and the number of classes (5) match your saved .pth file.")
        st.stop()

    # Set the model to evaluation mode (e.g., turns off dropout)
    model.eval()
    return model

# --- IMAGE PREPROCESSING ---
def preprocess_image(image_bytes):
    """
    Applies the necessary transformations to the uploaded image.
    
    *** IMPORTANT ***
    These transforms MUST match the transforms used during your model's training.
    The values below are standard for ImageNet, but may need adjustment.
    """
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        transformed_image = transform(image)
        
        # Add a batch dimension (e.g., [3, 224, 224] -> [1, 3, 224, 224])
        batch_image = transformed_image.unsqueeze(0)
        return batch_image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# --- STREAMLIT APP INTERFACE ---
st.set_page_config(page_title="Diabetic Retinopathy Detection", layout="wide")
st.title("👁️ Diabetic Retinopathy Classification (ResNet18)")
st.write("Upload a retinal scan image to classify its DR stage.")

# Load the model
model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose a retina image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Get image bytes
    image_bytes = uploaded_file.getvalue()
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    preprocessed_img = preprocess_image(image_bytes)
    
    if preprocessed_img is not None:
        # Make a prediction
        try:
            with torch.no_grad(): # Turn off gradients for inference
                output = model(preprocessed_img)
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # Get the top prediction
                confidence, predicted_index = torch.max(probabilities, 1)
                
                predicted_class_name = CLASS_NAMES[predicted_index.item()]
                confidence_percent = confidence.item() * 100

            # Display the result
            with col2:
                st.subheader("Prediction Result")
                st.success(f"**Diagnosis:** {predicted_class_name}")
                st.info(f"**Confidence:** {confidence_percent:.2f}%")
                
                # Optional: Display probabilities
                st.write("Full Probabilities:")
                st.dataframe({name: f"{prob*100:.2f}%" for name, prob in zip(CLASS_NAMES, probabilities[0])},
                             column_config={"value": "Probability"})

        except Exception as e:
            st.error(f"Error during prediction: {e}")
