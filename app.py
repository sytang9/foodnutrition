"""Main Streamlit application for Nutrition5k XAI."""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import requests

from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision import transforms

from config import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, TARGET_COLS, DEVICE,
    CF_STEPS, CF_LR, CF_LAM_L2, CF_LAM_TV, CF_GRAD_CLIP,
    OVERLAY_ALPHA, HEATMAP_CMAP
)
from utils import (
    _normalize_imagenet, _denormalize_imagenet, predict_raw,
    GradCAM, _overlay_heatmap, _gradcam_like_from_delta, _normalize_map_noisy,
    counterfactual_toward_target_calories
)

# Units for nutrition outputs
UNIT_MAP = {
    'calories': 'kcal',
    'mass': 'g',
    'fat': 'g',
    'carb': 'g',
    'protein': 'g',
}


# ============================================================================
# Page Config & Caching
# ============================================================================

st.set_page_config(page_title="Nutrition5k XAI", layout="wide")

@st.cache_resource
def load_model_and_checkpoint():
    """Load model and checkpoint from disk. If the checkpoint is missing and a MODEL_URL
    is set in Streamlit secrets, attempt to download it automatically.
    """
    ckpt_path = Path('convnext_tiny_nutrition5k_bestperdish.pt')

    if not ckpt_path.exists():
        model_url = st.secrets.get("https://huggingface.co/closear/foodnutritionperdish/resolve/main/convnext_tiny_nutrition5k_bestperdish.pt?download=true")
        if model_url:
            try:
                st.info("Downloading model checkpoint from configured MODEL_URL...")
                resp = requests.get(model_url, stream=True, timeout=60)
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(ckpt_path, 'wb') as f:
                    if total == 0:
                        f.write(resp.content)
                    else:
                        progress = st.progress(0)
                        dl = 0
                        for chunk in resp.iter_content(chunk_size=1024*1024):
                            if not chunk:
                                break
                            f.write(chunk)
                            dl += len(chunk)
                            progress.progress(min(100, int(dl * 100 / total)))
                st.success("Model downloaded.")
            except Exception as e:
                st.error(f"Failed to download model from MODEL_URL: {e}")
                st.stop()
        else:
            st.error(f"❌ Model checkpoint not found: {ckpt_path.resolve()}\nYou can set MODEL_URL in Streamlit Secrets to enable auto-download, or upload the model to the repo.")
            st.stop()

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu')

    weights = ConvNeXt_Tiny_Weights.DEFAULT
    model = convnext_tiny(weights=weights)

    target_cols_ckpt = ckpt.get('target_cols', TARGET_COLS)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = torch.nn.Linear(in_features, len(target_cols_ckpt))

    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = model.to(DEVICE).eval()

    target_mean = ckpt.get('target_mean', torch.zeros(5))
    target_std = ckpt.get('target_std', torch.ones(5))

    return model, target_cols_ckpt, target_mean, target_std, ckpt


@st.cache_resource
def get_test_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def process_image(image: Image.Image, transform) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    img_tensor = transform(image)  # (3, H, W), normalized
    return img_tensor


# ============================================================================
# Main App
# ============================================================================

st.title("🍽️ Nutrition5k XAI Explorer")
st.markdown("""
This application demonstrates **Explainable AI (XAI)** techniques for food image analysis.
- **Tab 1**: Upload an image and get Grad-CAM feature importance for the predicted nutrition value
- **Tab 2**: Upload an image and generate counterfactuals with visualizations
""")

# Load model
model, target_cols, target_mean, target_std, ckpt = load_model_and_checkpoint()
transform = get_test_transform()

# Determine calories index
cal_idx = target_cols.index('calories') if 'calories' in target_cols else 0

# Create tabs
tab1, tab2 = st.tabs(["Grad-CAM (Feature Importance)", "Counterfactuals (What-If Analysis)"])

# ============================================================================
# TAB 1: Grad-CAM
# ============================================================================
with tab1:
    st.header("Grad-CAM Feature Importance")
    st.markdown("""
    Upload a food image and select which nutrition output to explain.
    The heatmap shows which regions of the image most influence the model's prediction.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'], key='gradcam_upload')
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            st.subheader("Uploaded Image")
            st.image(image, width='stretch')
            
            # Get prediction
            with st.spinner("🔄 Predicting..."):
                x_norm = process_image(image, transform).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred_raw = predict_raw(model, x_norm, target_mean.to(DEVICE), target_std.to(DEVICE))[0].cpu().numpy()
            
            # Show all 5 predictions
            st.subheader("Predictions")
            pred_dict = {col: float(pred_raw[i]) for i, col in enumerate(target_cols)}
            for col, val in pred_dict.items():
                unit = UNIT_MAP.get(col, '')
                st.metric(label=col.capitalize(), value=f"{val:.1f} {unit}")
            
            # Let user select which output to explain
            selected_output = st.selectbox(
                "Select output to explain with Grad-CAM:",
                target_cols,
                index=cal_idx
            )
            selected_idx = target_cols.index(selected_output)
            
            if st.button("Generate Grad-CAM"):
                with st.spinner("⏳ Computing Grad-CAM..."):
                    try:
                        # Initialize Grad-CAM explainer
                        cam_explainer = GradCAM(model, target_layer=model.features[-1])
                        
                        x_norm_req = x_norm.clone().requires_grad_(True)
                        cam_map = cam_explainer(x_norm_req, target_index=selected_idx)
                        cam_up = torch.nn.functional.interpolate(
                            cam_map, size=x_norm.shape[-2:], mode='bilinear', align_corners=False
                        )[0, 0].detach().cpu().numpy()
                        
                        # Denormalize for display
                        x01 = _denormalize_imagenet(x_norm.detach().cpu()).clamp(0, 1)[0].permute(1, 2, 0).numpy()
                        
                        # Create overlay
                        heat_rgb = cm.jet(cam_up)[..., :3]
                        overlay = (1.0 - OVERLAY_ALPHA) * x01 + OVERLAY_ALPHA * heat_rgb
                        overlay = np.clip(overlay, 0, 1)
                        
                        # Display results
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.image(x01, caption="Original Image", width='stretch')
                        
                        with col_b:
                            st.image(overlay, caption="Grad-CAM Overlay", width='stretch')
                        
                        with col_c:
                            fig, ax = plt.subplots(figsize=(6, 6))
                            ax.imshow(cam_up, cmap='jet')
                            ax.set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
                            ax.axis('off')
                            st.pyplot(fig)
                        
                        unit = UNIT_MAP.get(selected_output, '')
                        st.success(f"✅ Grad-CAM computed for **{selected_output}** (prediction: {pred_dict[selected_output]:.1f} {unit})")
                        
                        cam_explainer.remove()
                    except Exception as e:
                        st.error(f"❌ Error computing Grad-CAM: {e}")


# ============================================================================
# TAB 2: Counterfactuals
# ============================================================================
with tab2:
    st.header("Counterfactual Analysis (What-If)")
    st.markdown("""
    Upload a food image and generate counterfactuals: minimal pixel-space changes that would move
    the model's calorie prediction toward a target value. This helps understand which image regions
    most influence calorie predictions.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file_cf = st.file_uploader("Choose an image (JPG, PNG)", type=['jpg', 'jpeg', 'png'], key='cf_upload')
        
        if uploaded_file_cf is not None:
            image_cf = Image.open(uploaded_file_cf).convert('RGB')
            
            st.subheader("Uploaded Image")
            st.image(image_cf, width='stretch')
            
            # Get initial prediction
            with st.spinner("🔄 Predicting..."):
                x_norm_cf = process_image(image_cf, transform).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred_raw_cf = predict_raw(model, x_norm_cf, target_mean.to(DEVICE), target_std.to(DEVICE))[0].cpu().numpy()
            
            st.subheader("Current Prediction")
            st.metric("Calories", f"{float(pred_raw_cf[cal_idx]):.1f} kcal")
            
            # Set target calories
            st.subheader("Target Calories")
            target_cal = st.slider(
                "Set target calories:",
                min_value=100.0,
                max_value=800.0,
                value=float(target_mean[cal_idx]),
                step=10.0
            )
            
            # Threshold for masked image
            st.subheader("Masked Image Settings")
            threshold = st.slider(
                "Heatmap threshold (0-1):",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Pixels above this threshold are blackened in the masked image"
            )
            
            if st.button("Generate Counterfactual"):
                with st.spinner("⏳ Optimizing counterfactual... (may take a minute)"):
                    try:
                        x0_cpu, xcf_cpu, pred0, predcf = counterfactual_toward_target_calories(
                            model=model,
                            x_norm=x_norm_cf[0],  # Remove batch dimension
                            target_cal=target_cal,
                            target_mean=target_mean.to(DEVICE),
                            target_std=target_std.to(DEVICE),
                            cal_idx=cal_idx,
                            steps=CF_STEPS,
                            lr=CF_LR,
                            lam_l2=CF_LAM_L2,
                            lam_tv=CF_LAM_TV,
                            grad_clip=CF_GRAD_CLIP,
                        )
                        
                        img0_display = x0_cpu.permute(1, 2, 0).numpy()
                        img_cf_display = xcf_cpu.permute(1, 2, 0).numpy()
                        
                        # Compute delta
                        delta = np.abs(img_cf_display - img0_display)
                        delta_gray = delta.mean(axis=2)
                        
                        # Smooth heatmap for overlay
                        heat_overlay = _gradcam_like_from_delta(delta_gray, down_size=28, q_low=60.0, q_high=99.7, gamma=0.45)
                        overlay = _overlay_heatmap(img0_display, heat_overlay, cmap_name='jet', alpha=0.35)
                        
                        # Masked image
                        mask = heat_overlay > threshold
                        img_masked = img0_display.copy()
                        img_masked[mask] = 0
                        
                        # Raw heatmap (noisy, non-smoothed)
                        heat_noisy = _normalize_map_noisy(delta_gray, q_high=99.5)
                        
                        # Display 5-panel layout in 2 rows
                        st.subheader("Results")
                        
                        # Row 1: Original, Overlay, Counterfactual
                        col_1, col_2, col_3 = st.columns(3)
                        
                        with col_1:
                            st.image(img0_display, caption="Original Image", width='stretch')
                            st.text(f"Pred: {float(pred0[cal_idx]):.1f} kcal")
                        
                        with col_2:
                            st.image(overlay, caption="Vibrant Overlay", width='stretch')
                            st.text("(Smoothed heatmap)")
                        
                        with col_3:
                            st.image(img_cf_display, caption="Counterfactual", width='stretch')
                            st.text(f"Pred: {float(predcf[cal_idx]):.1f} kcal")
                        
                        # Row 2: Masked and Raw Change
                        col_4, col_5 = st.columns(2)
                        
                        with col_4:
                            st.image(img_masked, caption=f"Masked (threshold={threshold:.2f})", width='stretch')
                            pct_masked = 100 * mask.mean()
                            st.text(f"{pct_masked:.1f}% masked")
                        
                        with col_5:
                            # Display raw heatmap using matplotlib
                            fig_raw, ax_raw = plt.subplots(figsize=(6, 6))
                            ax_raw.imshow(heat_noisy, cmap='magma')
                            ax_raw.set_title('Raw Absolute Change\n(Noisy, Per-pixel)', fontsize=11, fontweight='bold')
                            ax_raw.axis('off')
                            st.pyplot(fig_raw)
                        
                        st.success(f"✅ Counterfactual generated! Calories: {float(pred0[cal_idx]):.1f} kcal → {float(predcf[cal_idx]):.1f} kcal (target: {target_cal:.1f} kcal)")
                    
                    except Exception as e:
                        st.error(f"❌ Error generating counterfactual: {e}")


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown("""
**About this application:**
- Model: ConvNeXt-Tiny (ImageNet pretrained) fine-tuned on Nutrition5k dataset
- Grad-CAM: Gradient-based feature importance visualization
- Counterfactuals: CEM-style pixel-space optimization with sigmoid parameterization
- Framework: PyTorch + Streamlit
""")