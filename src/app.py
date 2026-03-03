import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Topology Optimization AI", layout="wide")

# --- 1. SIDEBAR CONFIGURATION ---
st.sidebar.header("⚙️ Simulation Settings")

st.sidebar.markdown("**1. Boundary Conditions**")
scenario = st.sidebar.radio(
    "Select Load Configuration:",
    ("Cantilever Beam (Edge Load)", "Center-Loaded Bracket", "Custom Load Configuration")
)

# Helper to calc forces
def get_force_components(mag, angle_deg):
    angle_rad = np.radians(angle_deg)
    return mag * np.cos(angle_rad), mag * np.sin(angle_rad)

use_load_2 = False 

# Logic for Presets vs Custom
if scenario == "Cantilever Beam (Edge Load)":
    l1_x, l1_y, l1_ang = 63, 32, 270
    st.sidebar.info("Preset: Fixed root with downward tip load.")
elif scenario == "Center-Loaded Bracket":
    l1_x, l1_y, l1_ang = 32, 32, 270
    st.sidebar.info("Preset: Fixed root with center downward load.")
else:
    st.sidebar.markdown("**Primary Load Vector**")
    l1_x = st.sidebar.slider("X Position (0-63)", 5, 63, 63, key="l1x")
    l1_y = st.sidebar.slider("Y Position (0-63)", 0, 63, 32, key="l1y")
    l1_ang = st.sidebar.slider("Angle (Degrees)", 0, 360, 270, key="l1a")
    
    st.sidebar.markdown("---")
    
    use_load_2 = st.sidebar.checkbox("Enable Second Load")
    if use_load_2:
        st.sidebar.markdown("**Secondary Load Vector**")
        l2_x = st.sidebar.slider("Sec. X (0-63)", 5, 63, 32, key="l2x")
        l2_y = st.sidebar.slider("Sec. Y (0-63)", 0, 63, 63, key="l2y")
        l2_ang = st.sidebar.slider("Sec. Angle (Degrees)", 0, 360, 90, key="l2a")

fx1, fy1 = get_force_components(1.0, l1_ang)
if use_load_2:
    fx2, fy2 = get_force_components(1.0, l2_ang)
else:
    fx2, fy2 = 0.0, 0.0

st.sidebar.markdown("---")
st.sidebar.markdown("**2. Output Style**")
show_raw = st.sidebar.checkbox("Show Raw Density (Organic)", value=False,
    help="Shows the raw probability field (0.0-1.0) instead of a binary threshold.")


# --- 2. MAIN PAGE SETUP ---
st.title("AI-Augmented Structural Topology Optimization")

# THE CALLOUT
st.warning("<--- **Configure your Load Cases & Force Vectors in the Sidebar!**")

st.info("""
**System Overview:** This application demonstrates the architectural evolution of a deep learning surrogate model for structural topology optimization.

Traditional Finite Element Analysis (FEA) solvers rely on computationally expensive, iterative matrix inversions. This project replaces that bottleneck with neural networks, achieving near-instantaneous inference. 

Switch between the tabs below to explore the project's progression:
1. **Phase 1 (Baseline):** A deterministic U-Net model constrained to a single mass target.
2. **Phase 2 (Advanced):** A Pix2Pix Generative Adversarial Network (GAN) introducing a 4th input channel for continuous, parametric mass control.
""")

# --- 3. LOAD MODELS (Cached) ---
@st.cache(allow_output_mutation=True)
def load_models():
    # Load Phase 1 (3-Channel U-Net)
    model_p1 = tf.keras.models.load_model('models/64_angled_combined.keras', compile=False)
    # Load Phase 2 (4-Channel GAN)
    model_p2 = tf.keras.models.load_model('models/final_topology_generator.h5', compile=False)
    
    # Warm up both models
    _ = model_p1(np.zeros((1, 64, 64, 3), dtype=np.float32), training=False)
    _ = model_p2(np.zeros((1, 64, 64, 4), dtype=np.float32), training=False)
    
    return model_p1, model_p2

try:
    model_p1, model_p2 = load_models()
except Exception as e:
    st.error(f"Error loading models. Ensure files are in 'models/' folder.\n{e}")
    st.stop()

# --- 4. TABS ---
tab1, tab2 = st.tabs(["Phase 1: Fixed Mass (U-Net)", "Phase 2: Variable Mass (GAN)"])

# ==========================================
# TAB 1: PHASE 1
# ==========================================
with tab1:
    st.subheader("Phase 1: Deterministic U-Net Architecture")
    
    st.markdown("""
    **Architecture:** The initial phase utilized a baseline U-Net Convolutional Neural Network. It accepts a **3-channel input tensor** representing the boundary conditions and the physical load vectors. It is rigidly trained to output structures at exactly 40% volume fraction.
    """)
    
    # Technical Metrics
    st.markdown("#### Training Data & Specifications")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Dataset Size", "23,000 Samples")
    colB.metric("Grid Resolution", "64x64")
    colC.metric("Input Channels", "3")
    colD.metric("Target Mass", "Fixed (40%)")
    
    with st.expander("View Governing Physics (SIMP Formulation)"):
        st.markdown("""
        The ground-truth training data was generated using a custom SIMP FEA solver. The objective function minimizes compliance:
        """)
        st.latex(r"""
        \min_{\mathbf{x}} : c(\mathbf{x}) = \mathbf{U}^T \mathbf{K} \mathbf{U}
        """)
        st.markdown("Subject to volume constraint $V(\mathbf{x}) \leq f V_0$.")

    st.markdown("---")
    
    # Format Phase 1 Input
    p1_input = np.zeros((1, 64, 64, 3), dtype=np.float32)
    p1_input[0, :, 0, 0] = 1.0
    p1_input[0, l1_y, l1_x, 1] += fx1
    p1_input[0, l1_y, l1_x, 2] += fy1
    if use_load_2:
        p1_input[0, l2_y, l2_x, 1] += fx2
        p1_input[0, l2_y, l2_x, 2] += fy2
        
    if st.button("Run Phase 1 Optimization", key="btn_p1"):
        start_time = time.time()
        p1_pred = model_p1(p1_input, training=False)
        p1_raw = p1_pred[0, :, :, 0].numpy()
        inf_time_p1 = time.time() - start_time
        
        # Visualization Logic
        if show_raw:
            display_img = 1.0 - p1_raw 
            caption_text = "Raw Density Field (Organic)"
        else:
            display_img = 1.0 - (p1_raw > 0.4).astype(float)
            caption_text = "Optimized Topology (Fixed 40% VF)"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Domain & Loads")
            fig_in, ax_in = plt.subplots(figsize=(4,4))
            
            # --- VIS CHANGE: RESTORE BLACK SQUARE ---
            vis_in = np.ones((64, 64)) # Start with 1.0 (Black/Material)
            vis_in[:, 0] = 0.5         # Gray boundary
            # ----------------------------------------
            
            # cmap='gray_r' means 0=White, 1=Black
            ax_in.imshow(vis_in, cmap='gray_r', vmin=0, vmax=1)
            ax_in.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=3, width=0.8, color='red')
            if use_load_2: ax_in.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=3, width=0.8, color='cyan') # Cyan pops on black
            ax_in.axis('off')
            st.pyplot(fig_in)
            
        with col2:
            st.markdown(f"### {caption_text}")
            fig_out, ax_out = plt.subplots(figsize=(4,4))
            im = ax_out.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            ax_out.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=3, width=0.8, color='red')
            if use_load_2: ax_out.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=3, width=0.8, color='cyan')
            ax_out.axis('off')
            
            if show_raw:
                cbar = plt.colorbar(im, ax=ax_out, fraction=0.046, pad=0.04)
                cbar.set_label('Material Density')
                
            st.pyplot(fig_out)
            
        st.success(f"Phase 1 Inference Time: {inf_time_p1:.3f} seconds.")

# ==========================================
# TAB 2: PHASE 2
# ==========================================
with tab2:
    st.subheader("Phase 2: Conditional GAN (Pix2Pix) with Parametric Control")
    
    st.markdown("""
    **Architecture:** To overcome the rigid mass constraints of Phase 1, the architecture was upgraded to a Conditional Generative Adversarial Network (cGAN). The input tensor was expanded to **4 channels**, introducing the *Target Volume Fraction* as a dynamic variable.
    """)
    
    # Technical Metrics
    st.markdown("#### Training Data & Specifications")
    colE, colF, colG, colH = st.columns(4)
    colE.metric("Dataset Size", "11,000+ Parametric") 
    colF.metric("Grid Resolution", "64x64")
    colG.metric("Input Channels", "4")
    colH.metric("Target Mass", "Dynamic (20%-80%)")
    
    with st.expander("View Adversarial Loss Formulation"):
        st.markdown("The network relies on an adversarial penalty where the U-Net Generator ($G$) attempts to fool the PatchGAN Discriminator ($D$).")
        st.latex(r"""
        \mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log(1 - D(x, G(x, z)))]
        """)

    st.markdown("---")
    
    target_vf_percent = st.slider("Target Volume Fraction (%)", 20, 80, 40, step=5, key="vf_slider")
    target_vf = target_vf_percent / 100.0
    
    # Format Phase 2 Input
    p2_input = np.zeros((1, 64, 64, 4), dtype=np.float32)
    p2_input[0, :, 0, 0] = 1.0
    p2_input[0, l1_y, l1_x, 1] += fx1
    p2_input[0, l1_y, l1_x, 2] += fy1
    if use_load_2:
        p2_input[0, l2_y, l2_x, 1] += fx2
        p2_input[0, l2_y, l2_x, 2] += fy2
    p2_input[0, :, :, 3] = target_vf 
    
    if st.button("Run Phase 2 Optimization", key="btn_p2"):
        start_time = time.time()
        p2_pred = model_p2(p2_input, training=False)
        p2_raw = p2_pred[0, :, :, 0].numpy()
        inf_time_p2 = time.time() - start_time
        
        # Visualization Logic
        if show_raw:
            display_img = 1.0 - p2_raw
            caption_text = f"Raw Density ({target_vf_percent}% Target)"
        else:
            display_img = 1.0 - (p2_raw > 0.5).astype(float) 
            caption_text = f"Optimized Topology ({target_vf_percent}% VF)"
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Domain & Loads")
            fig_in2, ax_in2 = plt.subplots(figsize=(4,4))
            
            # --- VIS CHANGE: RESTORE BLACK SQUARE ---
            vis_in2 = np.ones((64, 64)) # Start with 1.0 (Black/Material)
            vis_in2[:, 0] = 0.5         # Gray boundary
            # ----------------------------------------
            
            ax_in2.imshow(vis_in2, cmap='gray_r', vmin=0, vmax=1)
            ax_in2.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=3, width=0.8, color='red')
            if use_load_2: ax_in2.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=3, width=0.8, color='cyan')
            ax_in2.axis('off')
            st.pyplot(fig_in2)
            
        with col4:
            st.markdown(f"### {caption_text}")
            fig_out2, ax_out2 = plt.subplots(figsize=(4,4))
            im2 = ax_out2.imshow(display_img, cmap='gray', vmin=0, vmax=1)
            ax_out2.arrow(l1_x, l1_y, fx1*10, fy1*10, head_width=3, width=0.8, color='red')
            if use_load_2: ax_out2.arrow(l2_x, l2_y, fx2*10, fy2*10, head_width=3, width=0.8, color='cyan')
            ax_out2.axis('off')
            
            if show_raw:
                cbar = plt.colorbar(im2, ax=ax_out2, fraction=0.046, pad=0.04)
                cbar.set_label('Material Density')

            st.pyplot(fig_out2)
            
        st.success(f"Phase 2 Inference Time: {inf_time_p2:.3f} seconds.")