"""
Brain Tumor Detection AI — Professional Streamlit Application
==============================================================
AI Research & Educational Prototype powered by MobileNetV2
Author: Brain Tumor AI Research Team
Model: models/mobilenet_two_stage_final_unweighted.keras
"""

import os
import io
import textwrap
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

import sys

# Ensure project root is available in sys.path for scripts module imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from scripts.gradcam_explainer import generate_gradcam_overlay, GradCAMError

# ==============================================================================
# 1. Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="NeuroLens · Brain MRI Classification Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. Inject Professional UI Stylesheet
# ==============================================================================

def load_css():
    """Load custom style.css from assets directory with fallback."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    css_path = os.path.join(script_dir, "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback minimal styling if css file is not found
        st.markdown(
            """
            <style>
            .stApp { background-color: #080C14; color: #94A3B8; }
            h1, h2, h3 { color: #F8FAFC !important; }
            </style>
            """,
            unsafe_allow_html=True
        )

load_css()

# ==============================================================================
# 3. Model Loading (Cached Resource)
# ==============================================================================

@st.cache_resource
def get_model():
    """Load the verified retrained MobileNetV2 unweighted candidate model."""
    model_filename = "mobilenet_two_stage_final_unweighted.keras"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "models", model_filename)
    if not os.path.exists(model_path):
        model_path = os.path.join("models", model_filename)
    return load_model(model_path)

model = get_model()

# ==============================================================================
# 4. Class Mappings & Visual Theme Design Tokens
# ==============================================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

result_colors = {
    "Glioma": "#F87171",      # Soft Coral / Red
    "Meningioma": "#FBBF24",  # Warm Amber
    "No Tumor": "#34D399",    # Teal / Emerald
    "Pituitary": "#A78BFA"    # Refined Violet
}

# ==============================================================================
# 5. Left Sidebar Component
# ==============================================================================

with st.sidebar:
    # Brand Identity Box
    st.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="sidebar-brand-icon">🧠</div>
            <div>
                <div class="sidebar-brand-title">NeuroLens <span style="font-size: 0.72rem; font-weight: 500; color: #94A3B8;">· Brain MRI Classification Workspace</span></div>
                <span class="sidebar-brand-badge">Research & Educational Prototype</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Classification Classes Section
    st.markdown('<div class="sidebar-section-header">TARGET CLASSIFICATION CLASSES</div>', unsafe_allow_html=True)
    for cls in class_names:
        col = result_colors[cls]
        st.markdown(
            f"""
            <div class="sidebar-class-item">
                <div class="sidebar-class-left">
                    <span class="sidebar-class-dot" style="background-color: {col};"></span>
                    <span>{cls}</span>
                </div>
                <span style="font-size: 0.72rem; color: #64748B;">Category</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # Model Specifications Card
    st.markdown('<div class="sidebar-section-header">MODEL SPECIFICATIONS</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-info-card">
            <div class="sidebar-info-row">
                <span class="sidebar-info-key">Architecture</span>
                <span class="sidebar-info-val">MobileNetV2</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-key">Training Pipeline</span>
                <span class="sidebar-info-val">2-Stage Transfer</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-key">Input Matrix</span>
                <span class="sidebar-info-val">224 × 224 × 3</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-key">Normalization</span>
                <span class="sidebar-info-val">[-1.0, +1.0]</span>
            </div>
            <div class="sidebar-info-row">
                <span class="sidebar-info-key">Status</span>
                <span class="sidebar-info-val" style="color: #34D399;">Active Verified</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # About / Research Scope
    st.markdown('<div class="sidebar-section-header">PROJECT SCOPE</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p style="font-size: 0.78rem; line-height: 1.55; color: #717F94; margin: 0;">
            This deep learning interface evaluates MRI scans for educational exploration and technical benchmarking. 
            Designed for non-clinical research evaluation.
        </p>
        """,
        unsafe_allow_html=True
    )

# ==============================================================================
# 6. Hero Section
# ==============================================================================

st.markdown(
    """
    <div class="hero-container" style="margin-top: 0.5rem;">
        <div class="hero-tag-badge">
            <span>●</span> TRANSFER LEARNING · DEEP LEARNING · IMAGE CLASSIFICATION
        </div>
        <h1 class="hero-title">NeuroLens <span style="font-size: 1.4rem; font-weight: 400; color: #94A3B8; vertical-align: middle;">· Brain MRI Classification Workspace</span></h1>
        <p class="hero-subtitle">
            See what the model sees in a brain MRI.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Quick System Stat Cards Strip
s1, s2, s3, s4 = st.columns(4)
stat_items = [
    ("ARCHITECTURE", "MobileNetV2"),
    ("CLASSES", "4 Categories"),
    ("INPUT MATRIX", "224 × 224 px"),
    ("SYSTEM STATUS", "Operational")
]

for col, (label, val) in zip((s1, s2, s3, s4), stat_items):
    with col:
        status_color = "#34D399" if "Operational" in val else "#F1F5F9"
        dot = "🟢 " if "Operational" in val else ""
        st.markdown(
            f"""
            <div class="spec-chip">
                <div class="spec-chip-label">{label}</div>
                <div class="spec-chip-value" style="color: {status_color};">{dot}{val}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==============================================================================
# 7. Non-Clinical Educational & Research Disclaimer Banner
# ==============================================================================

st.markdown(
    """
    <div class="disclaimer-banner">
        <span style="font-size: 1.25rem; line-height: 1;">⚠️</span>
        <div>
            <strong>Non-Clinical Research Notice:</strong> 
            This application is an educational and research prototype. It is not clinically validated for medical diagnosis. 
            Model scores do not represent diagnostic certainty. Clinical imaging must be interpreted by qualified medical professionals.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# 8. MRI Upload Workspace
# ==============================================================================

with st.container(border=True):
    st.markdown('<div class="sidebar-section-header">MRI SCAN INGESTION WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="margin-bottom: 12px; font-size: 0.88rem; color: #8C99AE;">
            Upload an axial, coronal, or sagittal brain MRI scan in <b>JPG, JPEG, or PNG</b> format for automated 4-class evaluation.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload MRI Scan",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ==============================================================================
# 9. Prediction & Inference Workspace
# ==============================================================================

if uploaded_file is not None:
    try:
        # File integrity and format verification
        uploaded_file.seek(0)
        img_check = Image.open(uploaded_file)
        img_check.verify()

        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        file_size_kb = len(file_bytes) / 1024.0

        uploaded_file.seek(0)
        img = Image.open(uploaded_file).convert("RGB")
        preview_img = img.copy()
        orig_w, orig_h = img.size

        with st.spinner("Processing MRI tensor and executing MobileNetV2 inference..."):
            # Preprocessing strictly using MobileNetV2 canonical preprocess_input [-1.0, 1.0]
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            # Model prediction execution
            raw_pred = model.predict(img_array, verbose=0)[0]
            prediction = np.nan_to_num(raw_pred, nan=0.0)
            pred_sum = float(np.sum(prediction))
            if pred_sum > 0:
                prediction = prediction / pred_sum
            else:
                prediction = np.array([0.25, 0.25, 0.25, 0.25])

            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class])
            predicted_name = class_names[predicted_class]
            accent_color = result_colors[predicted_name]

            st.toast("✅ Analysis completed successfully")

    except Exception as e:
        st.error(
            f"❌ Could not process uploaded image: {e}. "
            "Please ensure you upload a valid, uncorrupted JPG, JPEG, or PNG brain MRI image."
        )
        st.stop()

    # Determine confidence badge tier
    if confidence >= 0.80:
        conf_pill_html = '<span class="conf-pill-high">● High Confidence (&ge;80%)</span>'
    elif confidence >= 0.60:
        conf_pill_html = '<span class="conf-pill-med">● Moderate Confidence (60–80%)</span>'
    else:
        conf_pill_html = '<span class="conf-pill-low">● Low Confidence (&lt;60%)</span>'

    # Grad-CAM Visual Explainability Execution (Isolated with Controlled Error Handling)
    gradcam_result = None
    gradcam_error = None
    try:
        with st.spinner("Generating Grad-CAM visual explanation..."):
            gradcam_result = generate_gradcam_overlay(
                model=model,
                preprocessed_tensor=img_array,
                original_image=preview_img,
                class_idx=predicted_class,
                target_layer_name="out_relu",
                alpha=0.4,
                colormap="jet"
            )
    except Exception as cam_exc:
        gradcam_error = str(cam_exc)

    st.write("")

    # ==========================================
    # Two-Column Presentation Layout
    # ==========================================

    left_col, right_col = st.columns([1, 1.15], gap="large")

    # ----- Left Column: Uploaded Scan -----
    with left_col:
        with st.container(border=True):
            st.markdown('<div class="sidebar-section-header">INGESTED MRI SCAN</div>', unsafe_allow_html=True)
            st.image(preview_img, use_container_width=True)
            st.markdown(
                f"""
                <div class="image-meta-grid">
                    <div class="image-meta-item">
                        <div class="image-meta-label">Original Filename</div>
                        <div class="image-meta-val" style="word-break: break-all;">{uploaded_file.name}</div>
                    </div>
                    <div class="image-meta-item">
                        <div class="image-meta-label">File Size</div>
                        <div class="image-meta-val">{file_size_kb:.1f} KB</div>
                    </div>
                    <div class="image-meta-item">
                        <div class="image-meta-label">Native Dimensions</div>
                        <div class="image-meta-val">{orig_w} × {orig_h} px</div>
                    </div>
                    <div class="image-meta-item">
                        <div class="image-meta-label">Model Tensor</div>
                        <div class="image-meta-val">224 × 224 (RGB)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # ----- Right Column: Model Prediction -----
    with right_col:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-card-accent-bar" style="background-color: {accent_color};"></div>
                <div class="result-card-eyebrow">MODEL PREDICTION</div>
                <div class="result-card-title">{predicted_name}</div>
                <div class="result-score-container">
                    <span class="result-score-value">{confidence * 100:.2f}%</span>
                    <span class="result-score-label">Softmax Score</span>
                    {conf_pill_html}
                </div>
                <div style="font-size: 0.78rem; color: #64748B; margin-top: 10px; line-height: 1.45;">
                    Score reflects the model's highest softmax output. It indicates mathematical classification likelihood, not clinical certainty.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # Class Softmax Probability Bars
        with st.container(border=True):
            st.markdown('<div class="sidebar-section-header">MODEL SOFTMAX OUTPUT SCORES</div>', unsafe_allow_html=True)
            for cls, prob in zip(class_names, prediction):
                c_color = result_colors[cls]
                pct = prob * 100.0
                bar_width = min(max(pct, 0.5), 100.0)
                st.markdown(
                    f"""
                    <div class="prob-row">
                        <div class="prob-header">
                            <span class="prob-label">
                                <span class="sidebar-class-dot" style="background-color: {c_color};"></span>
                                {cls}
                            </span>
                            <span class="prob-value">{pct:.2f}%</span>
                        </div>
                        <div class="prob-bar-track">
                            <div class="prob-bar-fill" style="width: {bar_width:.2f}%; background-color: {c_color};"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.markdown(
                '<div style="font-size: 0.74rem; color: #64748B; margin-top: 6px;">'
                'Normalized output distribution across 4 target classes. Values represent model softmax probabilities, not calibrated clinical diagnostic certainties.'
                '</div>',
                unsafe_allow_html=True
            )

    st.write("")

    # ==========================================
    # Compact Technical Specifications & Analysis Summary
    # ==========================================

    with st.container(border=True):
        st.markdown('<div class="sidebar-section-header">ANALYSIS SUMMARY & SPECIFICATIONS</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="summary-grid">
                <div class="summary-cell">
                    <div class="summary-cell-title">Predicted Category</div>
                    <div class="summary-cell-value" style="color: {accent_color};">{predicted_name}</div>
                </div>
                <div class="summary-cell">
                    <div class="summary-cell-title">Top Model Score</div>
                    <div class="summary-cell-value">{confidence * 100:.2f}%</div>
                </div>
                <div class="summary-cell">
                    <div class="summary-cell-title">Deep Learning Architecture</div>
                    <div class="summary-cell-value">MobileNetV2 (Transfer Learning)</div>
                </div>
                <div class="summary-cell">
                    <div class="summary-cell-title">Input Dimensions & Channels</div>
                    <div class="summary-cell-value">224 × 224 × 3 (RGB)</div>
                </div>
                <div class="summary-cell">
                    <div class="summary-cell-title">Preprocessing Method</div>
                    <div class="summary-cell-value">MobileNetV2 preprocess_input [-1.0, 1.0]</div>
                </div>
                <div class="summary-cell">
                    <div class="summary-cell-title">Inference Engine Status</div>
                    <div class="summary-cell-value" style="color: #34D399;">Active · Verified Checkpoint</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # ==========================================
    # Prediction Distribution Visualization
    # ==========================================

    with st.container(border=True):
        st.markdown('<div class="sidebar-section-header">PREDICTION DISTRIBUTION VISUALIZATION</div>', unsafe_allow_html=True)

        chart_data = pd.DataFrame({
            "Category": class_names,
            "Softmax Score": prediction
        })

        fig = px.pie(
            chart_data,
            names="Category",
            values="Softmax Score",
            hole=0.68,
            color="Category",
            color_discrete_map=result_colors
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            pull=[0.02, 0.02, 0.05, 0.02],
            marker=dict(line=dict(color="#080C14", width=2.5))
        )

        fig.update_layout(
            height=360,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1", family="Inter, sans-serif"),
            legend=dict(
                font=dict(color="#94A3B8"),
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=10, r=10, t=10, b=30)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.write("")

    # ==========================================
    # Visual Explanation — Grad-CAM
    # ==========================================

    with st.container(border=True):
        st.markdown('<div class="sidebar-section-header">VISUAL EXPLANATION — GRAD-CAM</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 0.88rem; color: #8C99AE; margin-bottom: 16px;">'
            'Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the spatial receptive fields '
            'in the MRI scan that contributed most strongly to the model\'s classification.'
            '</div>',
            unsafe_allow_html=True
        )

        if gradcam_result is not None and "overlay_image" in gradcam_result:
            cam_col1, cam_col2 = st.columns(2, gap="large")
            with cam_col1:
                st.markdown(
                    '<div style="font-weight: 600; color: #F8FAFC; margin-bottom: 8px; font-size: 0.95rem;">'
                    'Original MRI'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(preview_img, use_container_width=True)
                st.caption(f"Original Uploaded Scan ({orig_w} × {orig_h} px)")

            with cam_col2:
                st.markdown(
                    '<div style="font-weight: 600; color: #F8FAFC; margin-bottom: 8px; font-size: 0.95rem;">'
                    'Grad-CAM Heatmap Overlay'
                    '</div>',
                    unsafe_allow_html=True
                )
                st.image(gradcam_result["overlay_image"], use_container_width=True)
                st.caption(f"Attribution Overlay · Target Class: {predicted_name} (Layer: out_relu, α = 0.4)")

            st.markdown(
                """
                <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 14px; line-height: 1.55; padding: 10px 14px; background-color: rgba(15, 23, 42, 0.6); border-radius: 8px; border-left: 3px solid #38BDF8;">
                    The Grad-CAM heatmap highlights image regions that influenced the model's prediction. 
                    It is an AI explainability visualization and does not represent tumour segmentation, 
                    exact tumour boundaries, or a clinical diagnosis.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            err_msg = gradcam_error or "Unknown explainability computation error"
            st.warning(
                f"⚠️ Visual explanation could not be generated: {err_msg}. "
                "The classification output remains valid, but the Grad-CAM heatmap could not be computed for this scan."
            )

    st.write("")

    # ==========================================
    # Clinical Advisory Warning
    # ==========================================

    st.markdown(
        """
        <div class="disclaimer-banner">
            <span style="font-size: 1.25rem; line-height: 1;">🩺</span>
            <div>
                <strong>Clinical Advisory:</strong> 
                This AI analysis is intended strictly for educational and research purposes. 
                AI confidence reflects model softmax output and is not a guarantee of diagnostic certainty. 
                A board-certified radiologist or neurologist must interpret all clinical imaging.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================
    # Prediction Report Downloads (TXT + PDF)
    # ==========================================

    def generate_pdf_report(report_str):
        """
        Generate a professional vector PDF document from the authoritative report text.
        Uses existing matplotlib dependency without introducing new packages.
        """
        buf = io.BytesIO()
        fig = plt.figure(figsize=(8.5, 11), dpi=300)
        fig.patch.set_facecolor('#FFFFFF')
        
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        # Top header banner
        header_box = FancyBboxPatch(
            (0.06, 0.88), 0.88, 0.08,
            boxstyle="round,pad=0.015,rounding_size=0.015",
            facecolor="#0F172A", edgecolor="none"
        )
        ax.add_patch(header_box)
        
        ax.text(0.09, 0.932, "NeuroLens", fontsize=15, fontweight="bold", color="#38BDF8", va="center")
        ax.text(0.245, 0.932, "·  Brain MRI Classification Workspace", fontsize=11, color="#94A3B8", va="center")
        ax.text(0.09, 0.902, "BRAIN TUMOR DETECTION REPORT", fontsize=12, fontweight="bold", color="#F8FAFC", va="center")
        ax.text(0.91, 0.902, "RESEARCH PROTOTYPE", fontsize=8.5, fontweight="bold", color="#FBBF24", ha="right", va="center")
        
        # Parse lines from the authoritative report string
        lines = [line.strip() for line in report_str.strip().split("\n")]
        
        pred_val = ""
        conf_val = ""
        model_val = ""
        dim_val = ""
        prep_val = ""
        status_val = ""
        probs = []
        disclaimer_lines = []
        
        curr_section = None
        for line in lines:
            if line.startswith("===") or not line:
                continue
            if line.endswith(":"):
                curr_section = line[:-1]
                continue
            if curr_section == "Prediction":
                pred_val = line
            elif curr_section == "Confidence":
                conf_val = line
            elif curr_section == "Model":
                model_val = line
            elif curr_section == "Input Dimensions":
                dim_val = line
            elif curr_section == "Preprocessing":
                prep_val = line
            elif curr_section == "Probability Breakdown":
                if line.startswith("- "):
                    probs.append(line[2:])
            elif curr_section == "Status":
                status_val = line
            elif curr_section == "Clinical Disclaimer":
                disclaimer_lines.append(line)
                
        # Classification result card
        color_palette = {
            "Glioma": ("#FEF2F2", "#EF4444", "#B91C1C"),
            "Meningioma": ("#FFFBEB", "#F59E0B", "#B45309"),
            "No Tumor": ("#ECFDF5", "#10B981", "#047857"),
            "Pituitary": ("#F5F3FF", "#8B5CF6", "#6D28D9")
        }
        card_bg, border_col, text_col = color_palette.get(pred_val, ("#F8FAFC", "#94A3B8", "#0F172A"))
        
        pred_box = FancyBboxPatch(
            (0.06, 0.74), 0.88, 0.11,
            boxstyle="round,pad=0.015,rounding_size=0.012",
            facecolor=card_bg, edgecolor=border_col, linewidth=1.5
        )
        ax.add_patch(pred_box)
        
        ax.text(0.09, 0.825, "MODEL CLASSIFICATION RESULT", fontsize=9, fontweight="bold", color="#64748B", va="center")
        ax.text(0.09, 0.785, pred_val, fontsize=22, fontweight="bold", color=text_col, va="center")
        ax.text(0.09, 0.755, f"Confidence Score: {conf_val}", fontsize=11, fontweight="bold", color="#1E293B", va="center")
        
        # Technical Specifications Section
        ax.text(0.06, 0.705, "TECHNICAL SPECIFICATIONS", fontsize=10, fontweight="bold", color="#0F172A", va="center")
        ax.plot([0.06, 0.94], [0.695, 0.695], color="#CBD5E1", lw=1.0)
        
        specs = [
            ("Model Architecture", model_val or "MobileNetV2 (Retrained Two-Stage Unweighted)"),
            ("Input Dimensions", dim_val or "224 x 224 x 3 (RGB)"),
            ("Preprocessing Pipeline", prep_val or "MobileNetV2 preprocess_input [-1.0, 1.0]"),
            ("Execution Status", status_val or "Analysis Completed Successfully")
        ]
        
        y_spec = 0.67
        for label, val in specs:
            ax.text(0.09, y_spec, label, fontsize=9.5, fontweight="bold", color="#475569", va="center")
            ax.text(0.40, y_spec, val, fontsize=9.5, color="#0F172A", va="center")
            y_spec -= 0.026
            
        # Probability Breakdown Section
        y_spec -= 0.01
        ax.text(0.06, y_spec, "PROBABILITY BREAKDOWN", fontsize=10, fontweight="bold", color="#0F172A", va="center")
        y_spec -= 0.01
        ax.plot([0.06, 0.94], [y_spec, y_spec], color="#CBD5E1", lw=1.0)
        y_spec -= 0.025
        
        bar_colors = {
            "Glioma": "#F87171",
            "Meningioma": "#FBBF24",
            "No Tumor": "#34D399",
            "Pituitary": "#A78BFA"
        }
        
        for p_line in probs:
            parts = p_line.split(":")
            cls_name = parts[0].strip()
            val_str = parts[1].strip() if len(parts) > 1 else ""
            
            ax.text(0.09, y_spec, f"•  {cls_name}", fontsize=10, color="#1E293B", va="center")
            ax.text(0.40, y_spec, val_str, fontsize=10, fontweight="bold", color="#0F172A", va="center")
            
            try:
                pct_val = float(val_str.replace("%", "").strip())
                bar_len = (pct_val / 100.0) * 0.35
                if bar_len > 0.002:
                    bar_patch = FancyBboxPatch(
                        (0.55, y_spec - 0.006), bar_len, 0.012,
                        boxstyle="round,pad=0.002,rounding_size=0.004",
                        facecolor=bar_colors.get(cls_name, "#94A3B8"),
                        edgecolor="none"
                    )
                    ax.add_patch(bar_patch)
            except Exception:
                pass
                
            y_spec -= 0.026
            
        # Clinical Disclaimer Section
        y_spec -= 0.02
        disc_box = FancyBboxPatch(
            (0.06, y_spec - 0.12), 0.88, 0.12,
            boxstyle="round,pad=0.015,rounding_size=0.012",
            facecolor="#F8FAFC", edgecolor="#E2E8F0", linewidth=1.0
        )
        ax.add_patch(disc_box)
        
        ax.text(0.09, y_spec - 0.02, "CLINICAL DISCLAIMER & REGULATORY NOTICE", fontsize=9, fontweight="bold", color="#B45309", va="center")
        
        disc_text = " ".join(disclaimer_lines) if disclaimer_lines else (
            "This report is generated by an artificial intelligence research model and is strictly for "
            "educational and research use. It is not clinically validated for medical diagnosis. AI confidence "
            "reflects statistical classification probability, not diagnostic certainty. Clinical decisions "
            "must always be made by a qualified medical professional."
        )
        
        wrapped_lines = textwrap.wrap(disc_text, width=88)
        y_disc = y_spec - 0.045
        for wl in wrapped_lines:
            ax.text(0.09, y_disc, wl, fontsize=8, color="#64748B", va="center", style="italic")
            y_disc -= 0.018
            
        # Document Footer
        ax.plot([0.06, 0.94], [0.06, 0.06], color="#CBD5E1", lw=0.75)
        ax.text(0.06, 0.042, "NeuroLens · Brain MRI Classification Workspace  |  Authoritative Analysis Report", fontsize=8, color="#94A3B8", va="center")
        ax.text(0.94, 0.042, "File Format: PDF", fontsize=8, color="#94A3B8", ha="right", va="center")
        
        fig.savefig(buf, format="pdf", dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    report_content = f"""===============================
Brain Tumor Detection Report
===============================

Prediction:
{predicted_name}

Confidence:
{confidence * 100:.2f}%

Model:
MobileNetV2 (Retrained Two-Stage Unweighted)

Input Dimensions:
224 x 224 x 3 (RGB)

Preprocessing:
MobileNetV2 preprocess_input [-1.0, 1.0]

Probability Breakdown:
- Glioma: {prediction[0] * 100:.2f}%
- Meningioma: {prediction[1] * 100:.2f}%
- No Tumor: {prediction[2] * 100:.2f}%
- Pituitary: {prediction[3] * 100:.2f}%

Status:
Analysis Completed Successfully

Clinical Disclaimer:
This report is generated by an artificial intelligence research model
and is strictly for educational and research use. It is not clinically
validated for medical diagnosis. AI confidence reflects statistical
classification probability, not diagnostic certainty. Clinical decisions
must always be made by a qualified medical professional.
"""

    pdf_bytes = generate_pdf_report(report_content)

    col_txt, col_pdf = st.columns([1, 1])
    with col_txt:
        st.download_button(
            label="📄 Download Prediction Report",
            data=report_content,
            file_name="Brain_Tumor_Report.txt",
            mime="text/plain",
            type="primary",
            use_container_width=True
        )
    with col_pdf:
        st.download_button(
            label="📑 Download PDF Report",
            data=pdf_bytes,
            file_name="Brain_Tumor_Report.pdf",
            mime="application/pdf",
            type="secondary",
            use_container_width=True
        )

else:
    # State when awaiting upload: Guided explanation cards
    with st.container(border=True):
        st.markdown(
            """
            <div style="text-align: center; padding: 26px 16px;">
                <div style="font-size: 38px; margin-bottom: 8px;">🧠</div>
                <h3 style="margin: 0 0 8px 0; font-size: 1.25rem;">No MRI Scan Uploaded Yet</h3>
                <p style="max-width: 520px; margin: 0 auto 16px auto; font-size: 0.88rem; color: #718096; line-height: 1.6;">
                    Upload an axial, coronal, or sagittal brain MRI scan using the file uploader above. 
                    The model will automatically process the image and output probability scores across all four categories.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="sidebar-section-header">TARGET CLASSIFICATION PROFILES</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        profiles = [
            ("Glioma", "#F87171", "Primary brain tumors originating in glial cells of the brain and spinal tissue."),
            ("Meningioma", "#FBBF24", "Typically benign tumors developing within the meningeal membranous layers."),
            ("No Tumor", "#34D399", "Normal healthy brain tissue without detectable tumor abnormalities."),
            ("Pituitary", "#A78BFA", "Tumors originating in the pituitary endocrine gland at the cranial base.")
        ]

        for col, (name, hex_col, desc) in zip((c1, c2, c3, c4), profiles):
            with col:
                st.markdown(
                    f"""
                    <div class="summary-cell" style="border-top: 3px solid {hex_col}; height: 100%;">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                            <span class="sidebar-class-dot" style="background-color: {hex_col};"></span>
                            <span style="font-weight: 700; color: #F8FAFC; font-size: 0.92rem;">{name}</span>
                        </div>
                        <p style="font-size: 0.78rem; line-height: 1.5; color: #8B98AD; margin: 0;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ==============================================================================
# 10. Footer Component
# ==============================================================================

st.markdown(
    """
    <div class="footer-container">
        <b>NeuroLens</b> · Brain MRI Classification Workspace · AI Research & Educational Prototype<br>
        Powered by <b>MobileNetV2 (Retrained)</b> · TensorFlow · Streamlit<br>
        <span style="font-size: 0.74rem; color: #4B5563;">Not clinically validated for diagnostic decisions.</span>
    </div>
    """,
    unsafe_allow_html=True
)
