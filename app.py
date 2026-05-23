import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys
from PIL import Image

# Dynamically inject the parent directory into sys.path to guarantee local module resolution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import IMAGE_TRANSFORM
from model import MultimodalHousingPriceModel

# Set page configuration for a premium look
st.set_page_config(
    page_title="Multimodal House Price Estimator",
    page_icon="🏡",
    layout="wide"
)

MODEL_WEIGHTS_PATH = "multimodal_house_model.pth"
SCALER_PATH = "tabular_scaler.joblib"
ZIP_PATH = "zip_list.joblib"

@st.cache_resource
def load_prediction_model(weights_path: str, scaler_path: str, zip_path: str):
    """
    Loads and caches the scaler, zipcode vocabulary list, and custom multimodal model weights.
    Returns: (scaler, zip_list, model)
    """
    if not os.path.exists(scaler_path) or not os.path.exists(weights_path) or not os.path.exists(zip_path):
        return None, None, None
        
    try:
        # Load fitted StandardScaler and Zipcode vocabulary list
        scaler = joblib.load(scaler_path)
        zip_list = joblib.load(zip_path)
        
        # Reconstruct the model structure (dimension = 3 numeric + len(zip_list))
        model = MultimodalHousingPriceModel(tabular_input_dim=3 + len(zip_list))
        
        # Load state dictionary parameters on CPU
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()
        
        return scaler, zip_list, model
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

def main():
    st.title("🏡 Multimodal House Price Estimator")
    st.markdown(
        """
        *Predicting property values using a PyTorch Late Fusion network combining front-facing home images and structured listing details.*
        
        This application uses a pre-trained **MobileNetV2 CNN** to extract visual features from house photographs and combines them with tabular features in a neural regressor to estimate market valuations.
        """
    )
    
    # 1. Load scaler, zipcode vocabulary list, and model pipelines
    scaler, zip_list, model = load_prediction_model(MODEL_WEIGHTS_PATH, SCALER_PATH, ZIP_PATH)
    
    if scaler is None or zip_list is None or model is None:
        st.warning("⚠️ **Model Weights, Scaler, or Zipcode Vocabulary Not Found!**")
        st.info(
            f"""
            To run this multimodal dashboard:
            1. **Train your model on Kaggle** using our provided `multimodal_housing_price.ipynb` notebook.
            2. **Download** the generated files:
               * `multimodal_house_model.pth` (Model weights)
               * `tabular_scaler.joblib` (StandardScaler artifact)
               * `zip_list.joblib` (Zipcode vocabulary list)
            3. **Upload/Place** them directly into your GitHub repository root or local project folder:
               * `{os.path.abspath(MODEL_WEIGHTS_PATH)}`
               * `{os.path.abspath(SCALER_PATH)}`
               * `{os.path.abspath(ZIP_PATH)}`
            """
        )
        return

    st.markdown("---")
    
    # 2. Input panels (Layout: left inputs, right image uploader)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Home Characteristics (Tabular Modality)")
        bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.slider("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        area = st.number_input("Area (Square Feet)", min_value=300, max_value=15000, value=1800, step=50)
        zipcode = st.selectbox("Property Zipcode", zip_list)
        
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Estimate Valuation", type="primary", use_container_width=True)

    with col2:
        st.markdown("#### 📸 Frontal Image (Visual Modality)")
        uploaded_file = st.file_uploader(
            "Upload a front-facing photograph of the house (JPEG/PNG)", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Display uploaded image in real-time
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded House Image", use_container_width=True)
        else:
            st.caption("💡 *Tip: Upload a clean, frontal photo of a home to enable multimodal feature extraction.*")

    # 3. Model Inference pass
    if predict_button:
        if uploaded_file is None:
            st.warning("⚠️ Please upload a frontal photo of the house to run the visual modality!")
            return
            
        with st.spinner("Processing modalities and predicting price..."):
            try:
                # --- Visual Processing ---
                # Load image, apply torchvision transforms, and add batch dimension (1, 3, 224, 224)
                img = Image.open(uploaded_file).convert("RGB")
                img_tensor = IMAGE_TRANSFORM(img).unsqueeze(0)
                
                # --- Tabular Processing ---
                # Scale numerical features using loaded fitted StandardScaler
                num_data = np.array([[bedrooms, bathrooms, area]], dtype=np.float32)
                scaled_num = scaler.transform(num_data)[0]
                
                # One-hot encode Zipcode based on training features
                zip_vec = np.zeros(len(zip_list), dtype=np.float32)
                if zipcode in zip_list:
                    zip_idx = zip_list.index(zipcode)
                    zip_vec[zip_idx] = 1.0
                    
                # Concatenate features into single tabular vector and add batch dimension
                tab_vec = np.concatenate([scaled_num, zip_vec])
                tab_tensor = torch.tensor(tab_vec, dtype=torch.float32).unsqueeze(0)
                
                # --- Late Fusion Inference ---
                with torch.no_grad():
                    predicted_price_tensor = model(img_tensor, tab_tensor)
                    predicted_price = predicted_price_tensor.item()
                    
                # 4. Display Results
                st.markdown("---")
                st.subheader("📊 Model Estimation")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    # Highlight price card with a premium visual banner
                    st.success(f"### Estimated Market Price: **${predicted_price:,.2f}**")
                
                with res_col2:
                    st.markdown("#### Feature Contributions")
                    st.markdown(f"👤 **Space Density:** {area / bedrooms:.1f} sq.ft per bedroom")
                    st.markdown(f"🚿 **Convenience Ratio:** {bedrooms / bathrooms:.1f} bedrooms per bathroom")
                    st.markdown(f"📍 **Location Sector:** Zipcode {zipcode}")

            except Exception as e:
                st.error(f"Failed to execute multimodal prediction pipeline: {e}")

if __name__ == "__main__":
    main()
