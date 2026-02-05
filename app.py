import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Chest X-ray Diagnostic",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Multi-Modal Thoracic Pathology Detector")
st.markdown("""
This AI model analyzes Chest X-rays alongside patient clinical data (Age, Gender, View) 
to provide a multi-modal diagnostic suggestion.
""")

# --- 2. MODEL DOWNLOAD & LOADING ---
# If using OneDrive, follow the "Embed -> Change 'embed' to 'download'" trick for this URL
DIRECT_DOWNLOAD_URL = 'https://drive.google.com/file/d/1qKXaBm2-rUurSIPMFNXj_AeG20E9mm-6/view?usp=sharing'
MODEL_PATH = 'nih_chest_xray_multi_modal.keras'

@st.cache_resource
def load_multi_modal_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI Model from Cloud... Please wait."):
            try:
                # This downloads the file directly to the Streamlit server
                gdown.download(DIRECT_DOWNLOAD_URL, MODEL_PATH, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    
    return tf.keras.models.load_model(MODEL_PATH)

model = load_multi_modal_model()

# --- 3. SIDEBAR: CLINICAL METADATA ---
st.sidebar.header("📋 Patient Clinical Data")
st.sidebar.info("The AI uses this data to contextualize the X-ray scan.")

age = st.sidebar.slider("Patient Age", 0, 100, 45)
gender = st.sidebar.selectbox("Patient Gender", ["Male", "Female"])
view_pos = st.sidebar.selectbox("X-ray View Position", ["PA (Posteroanterior)", "AP (Anteroposterior)"])

# Preprocess Metadata (Must match your Training Scaling exactly!)
# Age: (Age - Mean) / StdDev | Gender: M=0, F=1 | View: PA=0, AP=1
age_scaled = (age - 46.8) / 16.6
gender_bin = 1 if gender == "Female" else 0
view_bin = 1 if "AP" in view_pos else 0

# Create the metadata vector for the model
metadata_vector = np.array([[age_scaled, gender_bin, view_bin]], dtype='float32')

# --- 4. MAIN INTERFACE: IMAGE UPLOAD ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload X-ray Scan")
    uploaded_file = st.file_uploader("Choose a PNG or JPG image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption='Uploaded Chest X-ray', use_column_width=True)

# --- 5. PREDICTION LOGIC ---
with col2:
    st.subheader("🔬 AI Analysis Results")
    
    if uploaded_file is not None and model is not None:
        if st.button('Analyze Multi-Modal Data'):
            with st.spinner('Fusing Vision and Clinical Data...'):
                # 1. Preprocess Image
                img_resized = img.resize((224, 224))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0).astype('float32')

                # 2. RUN PREDICTION (Multi-Input)
                # We pass a list: [Image_Branch_Input, Metadata_Branch_Input]
                preds = model.predict([img_array, metadata_vector])[0]

                # 3. Format Labels
                all_labels = [
                    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                    'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 'Nodule',
                    'Pleural_Thickening', 'Pneumonia', 'Pneumothorax', 'Hernia', 'No Finding'
                ]

                # 4. Display Top 3 Findings
                st.write("### Top Findings:")
                sorted_indices = np.argsort(preds)[::-1]
                
                for i in range(3):
                    idx = sorted_indices[i]
                    label = all_labels[idx]
                    confidence = preds[idx] * 100
                    
                    st.write(f"**{label}**")
                    st.progress(int(confidence))
                    st.write(f"Confidence: {confidence:.2f}%")
                    st.divider()

    else:
        st.info("Please upload an X-ray image and click 'Analyze' to see results.")

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Disclaimer: This tool is for educational/research purposes only and is not a substitute for professional medical advice.")