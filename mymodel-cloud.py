import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os


# the page configuration
st.set_page_config(page_title="Stroke CT Triage System", layout="centered")

# header
st.title("🧠 Automated Real-Time Stroke Triage System")
st.markdown("Please upload Brain CT slice to run the **EfficientNet-B0 + SE** model in real-time.")
st.markdown("This system is for Education Purpose Only. This is not real medical software on real world diagnosis.")
st.markdown("---")


# Load the proposed model 
@st.cache_resource(show_spinner="Loading AI weights into memory...")
def load_stroke_model():
    model_path = 'best_stroke_model.keras'
    if not os.path.exists(model_path):
        st.error(f"Critical Error: '{model_path}' not found in the repository. Please ensure it is uploaded to GitHub.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_stroke_model()

# Main Application GUI
if model is not None:
    uploaded_file = st.file_uploader("Upload a Brain CT Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Scan")
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Patient CT Slice", use_container_width=True)

        with col2:
            st.subheader("AI Triage Results")
            with st.spinner("Analyzing scan parameters..."):
                # Preprocessing
                img_resized = image.resize((224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_batch = np.expand_dims(img_array, axis=0)

                # Prediction
                prediction = model.predict(img_batch)
                class_labels = ['Bleeding', 'Ischemia', 'Normal'] 
                
                predicted_class_idx = np.argmax(prediction[0])
                predicted_label = class_labels[predicted_class_idx]
                confidence = float(prediction[0][predicted_class_idx] * 100)

                # Display Results
                if predicted_label == "Normal":
                    st.success(f"**Diagnosis:** {predicted_label}")
                elif predicted_label == "Ischemia":
                    st.warning(f"**Diagnosis:** {predicted_label} (Stroke Detected)")
                elif predicted_label == "Bleeding":
                    st.error(f"**Diagnosis:** {predicted_label} (Hemorrhagic Stroke Detected)")
                    
                st.metric(label="AI Confidence Score", value=f"{confidence:.2f}%")
                
                st.markdown("---")
                st.markdown("**Probability Breakdown:**")
                
                prob_df = pd.DataFrame({
                    'Condition': class_labels,
                    'Probability (%)': [float(p * 100) for p in prediction[0]]
                })
                st.bar_chart(prob_df.set_index('Condition'))