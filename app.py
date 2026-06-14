import os
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
from PIL import Image

# 1. SETUP EXECUTIVE PAGE BOUNDARIES & CONFIGURATIONS
st.set_page_config(
    page_title="VisionMetrics Pro | CNN Intelligence Studio",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. PREMIUM HIGH-CONTRAST LIGHT PALETTE (CSS INJECTION)
st.markdown("""
<style>
    /* Clean Soft-Gray Canvas Background */
    [data-testid="stAppViewContainer"] { 
        background-color: #f8fafc; 
    }
    
    /* Dark Slate Sidebar for a Professional Anchor */
    [data-testid="stSidebar"] { 
        background-color: #0f172a; 
        border-right: 1px solid #e2e8f0; 
    }
    
    /* Max Legibility Dark Charcoal Text for Labels and Explanations */
    h1, h2, h3, h4, h5, h6, .stMarkdown p, label { 
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important; 
        color: #0f172a !important; 
    }
    
    /* Clean Royal Blue to Ocean Blue Gradient Header */
    .header-hero {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
        padding: 40px; 
        border-radius: 16px; 
        margin-bottom: 30px; 
        text-align: center; 
        box-shadow: 0 10px 25px -5px rgba(30, 58, 138, 0.1);
        border-bottom: 5px solid #2563eb;
    }
    .header-hero h1 { color: #ffffff !important; font-size: 34px; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .header-hero p { color: #dbeafe !important; font-size: 15px; margin-top: 8px; font-weight: 400; opacity: 0.95; }
    
    /* Crisp White Background Cards to pop out of the Soft-Gray Base */
    .workspace-card { 
        background: #ffffff; 
        border: 1px solid #e2e8f0; 
        padding: 30px; 
        border-radius: 14px; 
        margin-bottom: 25px; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05); 
    }
    
    /* Section Separation Elements */
    .section-title {
        font-size: 20px; font-weight: 700; color: #1e3a8a !important; margin-bottom: 15px;
        display: flex; align-items: center; gap: 8px; border-bottom: 2px solid #f1f5f9; padding-bottom: 8px;
    }

    /* Light, Clean Telemetry Containers */
    div[data-testid="metric-container"] {
        background-color: #f1f5f9; 
        border: 1px solid #e2e8f0; 
        padding: 18px; 
        border-radius: 10px;
    }
    
    /* High-Visibility Cobalt Blue Inference Execution Controller */
    .stButton > button { 
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%); 
        color: white !important; 
        font-weight: 700; 
        font-size: 15px;
        width: 100%; 
        border-radius: 8px; 
        height: 48px; 
        border: none;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# 3. RUN APPLICATION HEADER DISPLAY
st.markdown("""
<div class="header-hero">
    <h1>VISIONMETRICS PRO • CNN INFERENCE STUDIO</h1>
    <p>Convolutional Neural Network Deep Learning Deployment Hub • Live Matrix Forward Pass Inspection</p>
</div>
""", unsafe_allow_html=True)

# 4. SECURE MODEL CACHE RESOURCING
@st.cache_resource
def load_cnn_model():
    model_path = "models/cnn_cifar10_model.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

cnn_model = load_cnn_model()

if cnn_model is None:
    st.error("🚨 Deployment Target Offline! Missing trained model matrix file. Execute 'python src/cnn_training.py' inside your core command prompt first.")
    st.stop()

# Target prediction class index array
cifar_classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
test_dir = "data/test"

# ==============================================================================
# SIDEBAR CONTROL DESIGN PANEL
# ==============================================================================
with st.sidebar:
    st.markdown("<h3 style='color: white !important;'>⚙️ Studio Engine Logs</h3>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color: #94a3b8 !important; font-size: 14px;'>Model Core Layer Spec:<br><strong style='color: white;'>CNN - Keras Sequential</strong></p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8 !important; font-size: 14px;'>Total Output Classes:<br><strong style='color: white;'>10 Categories</strong></p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p style='color: #94a3b8 !important; font-size: 13px;'>Execution Server Status: <span style='color: #10b981; font-weight: bold;'>ONLINE</span></p>", unsafe_allow_html=True)

# ==============================================================================
# MAIN WORKSPACE INTERFACE ARCHITECTURE
# ==============================================================================
st.markdown('<div class="workspace-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📊 Operational Overview & Telemetry Metrics</div>', unsafe_allow_html=True)

if os.path.exists(test_dir):
    # Detect subdirectories dynamically from local Kaggle extraction folder
    available_folders = [f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
    
    # Render layout selector widgets
    col_sel_class, col_sel_idx = st.columns(2)
    with col_sel_class:
        selected_folder = st.selectbox("🎯 Target Evaluation Class Folder:", available_folders)
    
    folder_path = os.path.join(test_dir, selected_folder)
    images_in_folder = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if images_in_folder:
        with col_sel_idx:
            img_idx = st.slider("🎞️ Dataset File Navigation Matrix Slider", 0, len(images_in_folder)-1, 0)
            
        target_img_name = images_in_folder[img_idx]
        full_img_path = os.path.join(folder_path, target_img_name)
        
        # Split layout cleanly between Image Analysis and Prediction Panels
        col_view_frame, col_pred_frame = st.columns([1, 2], gap="large")
        
        # ---- PANEL A: SHARP IMAGE RENDERING SYSTEM ----
        with col_view_frame:
            st.markdown("### 📷 High-Fidelity Target Image")
            img_pil = Image.open(full_img_path)
            
            # === THE BLURRY IMAGES FIX ===
            # Upscale the natively tiny 32x32 image using high-quality NEAREST-NEIGHBOR interpolation.
            # This locks down pixel borders so that the upscaled image looks crisp and clean.
            upscale_resolution = (256, 256)
            sharpened_display_image = img_pil.resize(upscale_resolution, resample=Image.NEAREST)
            
            # Display image cleanly utilizing the version-compatible layout configuration parameter
            st.image(sharpened_display_image, caption=f"File: {target_img_name} (Pixel-Perfect Sharp View)", use_column_width=True)
            
        # ---- PANEL B: INFRASTRUCTURE REAL-TIME PREDICTION ----
        with col_pred_frame:
            st.markdown("### 🔮 Real-Time Network Inference Pass")
            st.write("Clicking the execution controller triggers an active forward pass across the deep convolutional feature map structures.")
            st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("EXECUTE FORWARD INFERENCE PASS"):
                # Preprocess verification image matrices to conform with network input dimensions (32x32)
                img_resized = img_pil.resize((32, 32))
                img_array = np.array(img_resized) / 255.0
                
                # Check channel dimensions for grayscale safety mapping
                if img_array.shape[-1] != 3:
                    img_array = np.stack((img_array,)*3, axis=-1)
                    
                input_tensor = np.expand_dims(img_array, axis=0)
                
                # Calculate prediction pass weights
                predictions_matrix = cnn_model.predict(input_tensor, verbose=0)
                predicted_class_idx = np.argmax(predictions_matrix)
                confidence_score = np.max(predictions_matrix) * 100
                
                # Render high-contrast, easy-to-read result telemetry badges
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #10b981 0%, #059669 100%); padding: 18px; border-radius: 10px; color: white; margin-bottom: 15px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);">
                    <span style='font-size: 11px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; font-weight: bold;'>Primary Classification Outcome</span>
                    <h3 style='color: #ffffff !important; margin: 4px 0 0 0; font-size: 24px; font-weight:700;'>🎯 Class Result: {cifar_classes[predicted_class_idx].upper()}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #2563eb 0%, #1e40af 100%); padding: 18px; border-radius: 10px; color: white; margin-bottom: 25px; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);">
                    <span style='font-size: 11px; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; font-weight: bold;'>Inference Telemetry Metrics</span>
                    <h4 style='color: #ffffff !important; margin: 4px 0 0 0; font-size: 18px; font-weight:600;'>📊 Probability Confidence: {confidence_score:.2f}%</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution visualization
                st.markdown("##### Softmax Probability Distribution Vector Map:")
                chart_data = pd.DataFrame(predictions_matrix.T, index=cifar_classes, columns=["Activation Strength"])
                st.bar_chart(chart_data, y="Activation Strength", use_container_width=True)
    else:
        st.warning("Empty directory detected within selected categorical target partition folder.")
else:
    st.error("🚨 Layout Warning: Local data source structures tracking error. Ensure paths are rooted accurately in 'data/test'.")

st.markdown('</div>', unsafe_allow_html=True)