import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Topology Optimization AI", layout="wide")
st.title("AI-Augmented Topology Optimization (64x64)")
st.markdown("Instantly predict optimal structural layouts using a custom U-Net CNN.")

# --- 2. LOAD MODEL (CACHED) ---
# The @st.cache_resource decorator ensures the model only loads once, 
# preventing lag every time you change a slider.
@st.cache(allow_output_mutation=True)
def load_ai_model():
    return tf.keras.models.load_model('64_angled_grandmaster.keras')

try:
    model = load_ai_model()
    st.sidebar.success("Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"Error loading model. Ensure '64_angled_grandmaster.keras' is in the directory.\n{e}")
    st.stop()

# --- 3. SIDEBAR (USER INPUTS) ---
st.sidebar.header("Load Configurations")

# Function to get forces (borrowed from your validation script)
def get_force_components(mag, angle_deg):
    angle_rad = np.radians(angle_deg)
    return mag * np.cos(angle_rad), mag * np.sin(angle_rad)

# Load 1 Inputs
st.sidebar.subheader("Load 1")
l1_x = st.sidebar.slider("X Position (0-63)", 5, 63, 63, key="l1x")
l1_y = st.sidebar.slider("Y Position (0-63)", 0, 63, 32, key="l1y")
l1_ang = st.sidebar.slider("Angle (Degrees)", 0, 360, 270, key="l1a")

# Load 2 Inputs (Optional multi-load)
st.sidebar.subheader("Load 2 (Optional)")
use_load_2 = st.sidebar.checkbox("Enable Second Load")
if use_load_2:
    l2_x = st.sidebar.slider("X Position (0-63)", 5, 63, 32, key="l2x")
    l2_y = st.sidebar.slider("Y Position (0-63)", 0, 63, 63, key="l2y")
    l2_ang = st.sidebar.slider("Angle (Degrees)", 0, 360, 90, key="l2a")

# Threshold filter
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Density Threshold (Filter Blur)", 0.1, 0.9, 0.4)

#4. PREPARE AI INPUT TENSOR
# Initialize the 1x64x64x3 tensor
ai_input = np.zeros((1, 64, 64, 3))

# Channel 0: Fixed Boundary (Left Wall)
ai_input[0, :, 0, 0] = 1 

# Channel 1 & 2: Apply Load 1
fx1, fy1 = get_force_components(1.0, l1_ang)
ai_input[0, l1_y, l1_x, 1] += fx1
ai_input[0, l1_y, l1_x, 2] += fy1

# Channel 1 & 2: Apply Load 2 (if enabled)
if use_load_2:
    fx2, fy2 = get_force_components(1.0, l2_ang)
    ai_input[0, l2_y, l2_x, 1] += fx2
    ai_input[0, l2_y, l2_x, 2] += fy2

#5. RUN INFERENCE
if st.button("Generate Topology"):
    with st.spinner("AI is thinking..."):
        # Run the model
        prediction = model.predict(ai_input, verbose=0)
        
        # Apply threshold to get binary output
        binary_output = (prediction[0, :, :, 0] > threshold).astype(float)

        # Plotting
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Problem Definition (Input)")
            fig_in, ax_in = plt.subplots(figsize=(5,5))
            ax_in.imshow(ai_input[0, :, :, 0], cmap='Greys', alpha=0.3) # Show wall
            # Plot arrows for forces
            ax_in.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=2, color='red')
            if use_load_2:
                ax_in.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=2, color='blue')
            ax_in.set_title("Boundary Conditions & Loads")
            st.pyplot(fig_in)
            
        with col2:
            st.subheader("AI Optimized Structure (Output)")
            fig_out, ax_out = plt.subplots(figsize=(5,5))
            # cmap 'magma' or 'binary' works well. Inverted so solid is black.
            ax_out.imshow(1 - binary_output, cmap='gray', vmin=0, vmax=1) 
            ax_out.set_title(f"Inference Time: ~0.1s")
            st.pyplot(fig_out)