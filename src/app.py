import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Topology Optimization AI", layout="wide")
st.title("AI-Augmented Structural Topology Optimization")

st.info("""
**System Overview:** This application provides real-time structural topology optimization for mechanical components. It determines the most efficient material distribution within a given design space, maximizing stiffness while minimizing mass.

Rather than relying on computationally expensive finite element analysis (FEA) iterations, this system utilizes a trained deep learning surrogate model to predict the optimal structure. 
1. **Boundary Conditions:** The design domain is constrained at specified anchor points.
2. **Load Application:** Forces are applied at user-defined coordinates.

The model rapidly eliminates 60% of the non-essential material volume, outputting a high-strength, lightweight structural topology.
""")

with st.expander("Technical Specifications"):
    st.markdown("""
    * **Model Architecture:** U-Net Convolutional Neural Network
    * **Training Data Generation:** Custom Python Finite Element Analysis (FEA) solver 
    * **Performance:** ~0.2s inference time (up to 40x speedup vs. iterative solvers)
    * **Validation Accuracy:** 92.3% Intersection-over-Union (IoU)
    * **Constraints:** Volume fraction is fixed at **0.4 (40%)** to ensure consistent structural integrity.
    """)

st.markdown("---")

@st.cache(allow_output_mutation=True)
def load_ai_model():
    return tf.keras.models.load_model('models/64_angled_grandmaster.keras')

try:
    model = load_ai_model()
except Exception as e:
    st.error(f"Error loading model. Ensure '../models/64_angled_grandmaster.keras' is in the directory.\n{e}")
    st.stop()

st.subheader("Step 1: Define Boundary Conditions and Loads")

st.write("""
**Domain Configuration:**
The workspace represents a 64x64 discrete design domain.
* **Fixed Support:** The entire left boundary is fully constrained (zero displacement).
* **Applied Load:** Select a standard testing configuration or define custom coordinate loads using the sidebar.
""")

scenario = st.radio(
    "Select Load Configuration:",
    ("Cantilever Beam (Edge Load)", 
     "Center-Loaded Bracket",
     "Custom Load Configuration")
)

def get_force_components(mag, angle_deg):
    angle_rad = np.radians(angle_deg)
    return mag * np.cos(angle_rad), mag * np.sin(angle_rad)

use_load_2 = False 

if scenario == "Cantilever Beam (Edge Load)":
    st.success("Configuration applied: Downward force at the far right edge.")
    l1_x, l1_y, l1_ang = 63, 32, 270

elif scenario == "Center-Loaded Bracket":
    st.success("Configuration applied: Downward force at the center coordinate.")
    l1_x, l1_y, l1_ang = 32, 32, 270

else:
    st.write("**Define custom load parameters in the sidebar.**")
    st.sidebar.header("Custom Load Parameters")
    st.sidebar.subheader("Primary Load")
    l1_x = st.sidebar.slider("X Position (0-63)", 5, 63, 63, key="l1x")
    l1_y = st.sidebar.slider("Y Position (0-63)", 0, 63, 32, key="l1y")
    l1_ang = st.sidebar.slider("Angle (Degrees)", 0, 360, 270, key="l1a")
    
    st.sidebar.subheader("Secondary Load (Optional)")
    use_load_2 = st.sidebar.checkbox("Enable Second Load")
    if use_load_2:
        l2_x = st.sidebar.slider("X Position (0-63)", 5, 63, 32, key="l2x")
        l2_y = st.sidebar.slider("Y Position (0-63)", 0, 63, 63, key="l2y")
        l2_ang = st.sidebar.slider("Angle (Degrees)", 0, 360, 90, key="l2a")

threshold = 0.4 

ai_input = np.zeros((1, 64, 64, 3))
ai_input[0, :, 0, 0] = 1 

fx1, fy1 = get_force_components(1.0, l1_ang)
ai_input[0, l1_y, l1_x, 1] += fx1
ai_input[0, l1_y, l1_x, 2] += fy1

if use_load_2:
    fx2, fy2 = get_force_components(1.0, l2_ang)
    ai_input[0, l2_y, l2_x, 1] += fx2
    ai_input[0, l2_y, l2_x, 2] += fy2

st.markdown("---")

st.subheader("Step 2: Execute Optimization")

if st.button("Run Optimization"):
    with st.spinner("Computing optimal topology..."):
        
        # --- Start Timer ---
        start_time = time.time()
        
        # Run the model
        prediction = model.predict(ai_input, verbose=0)
        
        # --- Stop Timer ---
        end_time = time.time()
        inference_time = end_time - start_time
        
        binary_output = (prediction[0, :, :, 0] > threshold).astype(float)

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Initial Design Domain")
            st.caption("Left boundary constrained (grey). Vectors indicate applied loads.")
            
            fig_in, ax_in = plt.subplots(figsize=(5,5))
            
            visual_domain = np.zeros((64, 64)) 
            visual_domain[:, 0] = 0.5 
            
            ax_in.imshow(visual_domain, cmap='gray', vmin=0, vmax=1)
            
            ax_in.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=3, width=0.8, color='#ff3333') 
            if use_load_2:
                ax_in.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=3, width=0.8, color='#32CD32') 
                
            ax_in.axis('off') 
            st.pyplot(fig_in)
            
        with col2:
            st.markdown("### Optimized Topology")
            st.caption("Resulting structure at 40% volume fraction.")
            
            fig_out, ax_out = plt.subplots(figsize=(5,5))
            
            visual_out = 1.0 - binary_output
            visual_out[:, 0] = 0.5 
            
            ax_out.imshow(visual_out, cmap='gray', vmin=0, vmax=1) 
            
            ax_out.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=3, width=0.8, color='#ff3333')
            if use_load_2:
                ax_out.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=3, width=0.8, color='#32CD32')
                
            ax_out.axis('off') 
            st.pyplot(fig_out)
            
        # Dynamically display the actual inference time formatted to 3 decimal places
        st.success(f"Optimization completed successfully in {inference_time:.3f} seconds.")