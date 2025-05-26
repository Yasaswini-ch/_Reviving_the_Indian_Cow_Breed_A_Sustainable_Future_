

import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import logging

# --- Initialize Logging (Optional for standalone, but good practice) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# These class names must correspond to the output classes of your trained model, in the correct order.
CLASS_NAMES = ["Bacterial", "Fungal", "Healthy"]

# Information and advice for each predicted class
DISEASE_INFO_AND_ADVICE = {
    "Bacterial": {
        "common_examples": "Examples include Dermatophilosis (Rain Scald), Staphylococcal infections (Pyoderma).",
        "general_appearance": "Often present as pustules (pus-filled bumps), moist/oozing lesions, crusts, or abscesses. The skin might be inflamed and painful.",
        "possible_factors": "Skin injuries, prolonged wetness, poor hygiene, insect bites, or a compromised immune system can be predisposing factors.",
        "advice": [
            "**Consult a veterinarian immediately for accurate diagnosis and appropriate antibiotic treatment.**",
            "Isolate the affected animal to prevent potential spread to others.",
            "Keep the affected area clean and dry as much as possible (follow vet guidance).",
            "Improve overall hygiene in the animal's living environment.",
            "Do not apply any ointments or medications without veterinary prescription."
        ]
    },
    "Fungal": {
        "common_examples": "The most common is Dermatophytosis (Ringworm).",
        "general_appearance": "Typically appears as circular, scaly, hairless patches. Lesions may be slightly raised, crusted, and sometimes itchy. Can occur anywhere on the body.",
        "possible_factors": "Often spread by direct contact with infected animals or contaminated environments/equipment. Young, old, or immunocompromised animals are more susceptible. Humid conditions can favor growth.",
        "advice": [
            "**Consult a veterinarian for confirmation and appropriate antifungal treatment (topical and/or systemic).**",
            "Ringworm is zoonotic (can spread to humans) and highly contagious to other animals. Isolate the affected animal strictly.",
            "Thoroughly clean and disinfect the animal's environment, grooming tools, and any items it contacted.",
            "Wear gloves when handling the affected animal or cleaning its environment.",
            "Ensure good ventilation and dry conditions for all animals."
        ]
    },
    "Healthy": {
        "common_examples": "No signs of common bacterial or fungal infections detected in the image.",
        "general_appearance": "Skin appears clear, without unusual lesions, scaling, or significant hair loss in the area shown.",
        "possible_factors": "Good hygiene, balanced nutrition, and a stress-free environment contribute to healthy skin.",
        "advice": [
            "The AI suggests the skin in the image appears healthy based on its training.",
            "Continue good management practices and regular observation of all animals for any changes in skin health.",
            "Remember, this tool only analyzes the provided image. For a complete health assessment, always rely on regular veterinary check-ups."
        ]
    }
}

@st.cache_resource
def load_skin_disease_model():
    """Loads the pre-trained skin disease prediction model."""
    # Adjust this path if your model is located elsewhere relative to this script.
    # This assumes the script is run from the root of AnimalSkinDiseasePrediction_CNN
    # and the model is in AnimalSkinDiseasePrediction_CNN/model/model.h5
    model_path = os.path.join(os.getcwd(), 'model', 'model.h5')
    
    if not os.path.exists(model_path):
        logger.error(f"Disease prediction model not found at: {model_path}")
        st.error(f"🐛 Disease prediction model not found. Please ensure the model file is at: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info("Skin disease prediction model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading skin disease prediction model: {e}", exc_info=True)
        st.error(f"🐛 Error loading skin disease prediction model: {e}")
        return None

def preprocess_skin_image(img: Image.Image):
    """Preprocesses the uploaded image for model prediction."""
    # Resize to the model's expected input size (e.g., 150x150)
    img = img.resize((150, 150))
    # Convert PIL image to NumPy array
    arr = keras_image.img_to_array(img)
    # Expand dimensions to create a batch of 1
    arr = np.expand_dims(arr, 0)
    # Normalize pixel values (if your model was trained with normalized data)
    arr = arr / 255.0
    return arr

def render_disease_detector():
    """
    Renders the Streamlit UI for the cattle disease predictor.
    This function is designed to be callable from another Streamlit app.
    """
    st.title("🐄 Cattle Disease Predictor (Image-Based)") # Title can be set here or in the calling app
    st.markdown("Upload a clear close-up image of the affected skin or hoof area for a preliminary AI-based prediction.")
    st.warning("**Disclaimer:** This tool provides an initial suggestion and is **NOT** a substitute for professional veterinary diagnosis. Always consult a qualified veterinarian for any health concerns.", icon="⚠️")
    st.markdown("---")

    uploaded_file = st.file_uploader("Choose an image (JPG, PNG, JPEG)...", type=['png','jpg','jpeg'], key="disease_detector_uploader")
    
    if uploaded_file is None:
        st.info("Please upload an image of the affected area to get a prediction.")
        return

    try:
        # Display the uploaded image
        img_pil = Image.open(uploaded_file).convert("RGB") # Ensure image is RGB
        st.image(img_pil, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image and make prediction
        processed_tensor = preprocess_skin_image(img_pil)
        model = load_skin_disease_model()

        if model is None:
            # Error message already shown by load_skin_disease_model
            return

        with st.spinner("🧠 Analyzing image... Please wait."):
            predictions = model.predict(processed_tensor)
            
        # Get the class with the highest probability
        predicted_class_index = np.argmax(predictions[0])
        confidence_score = predictions[0][predicted_class_index]
        predicted_class_name = CLASS_NAMES[predicted_class_index]

        st.success(f"**Predicted Condition: {predicted_class_name}** ({confidence_score:.1%} confidence)")
        
        # Display descriptive information and advice
        if predicted_class_name in DISEASE_INFO_AND_ADVICE:
            info = DISEASE_INFO_AND_ADVICE[predicted_class_name]
            with st.container(border=True):
                st.markdown(f"##### General Information about Potential '{predicted_class_name}' Conditions:")
                if "common_examples" in info:
                    st.markdown(f"**Common Examples:** {info['common_examples']}")
                if "general_appearance" in info:
                    st.markdown(f"**General Appearance:** {info['general_appearance']}")
                if "possible_factors" in info:
                    st.markdown(f"**Possible Contributing Factors:** {info['possible_factors']}")
                
                st.markdown("##### Recommended Actions & Advice:")
                for point in info["advice"]:
                    if "**Consult a veterinarian immediately**" in point:
                        st.error(point, icon="👩‍⚕️") # Emphasize vet consultation
                    elif predicted_class_name == "Healthy" and "AI suggests" in point:
                         st.info(point, icon="💡")
                    else:
                        st.markdown(f"- {point}")
        
        # Reinforce disclaimer based on prediction
        if predicted_class_name != "Healthy":
            st.error("**Important:** This AI prediction is for informational purposes only. It is crucial to consult a qualified veterinarian for an accurate diagnosis and appropriate treatment plan.", icon="⚠️")
        else:
            st.info("While the AI suggests the area in the image appears healthy, continue regular observation and consult your vet for routine health checks or if you notice any changes in your animal's condition.", icon="ℹ️")

    except Exception as e:
        logger.error(f"Error during disease prediction process: {e}", exc_info=True)
        st.error(f"An error occurred during image processing or prediction: {e}")
        st.markdown("Please ensure the uploaded file is a valid image (PNG, JPG, JPEG). If the problem persists, the image might be corrupted or in an unsupported format.")

# This part allows the script to be run as a standalone Streamlit app for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Cattle Disease Predictor Test", layout="centered")
    render_disease_detector()