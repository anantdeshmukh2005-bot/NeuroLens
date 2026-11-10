"""
scripts/generate_architecture_diagram.py
=========================================
Generates a professional, high-resolution technical system architecture diagram
for NeuroLens (assets/neuroLens_architecture.png and assets/neuroLens_architecture.svg).
Features:
- Offline Model Development Pipeline (Two-Stage Transfer Learning, Leakage-Free Audit)
- Real-Time Inference & Explainable AI Pipeline (MobileNetV2, out_relu Grad-CAM, Reporting)
- Medical-grade dark slate theme (#0B0F17) with clear visual hierarchy and legible typography.
"""

import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

PNG_PATH = os.path.join(ASSETS_DIR, "neuroLens_architecture.png")
SVG_PATH = os.path.join(ASSETS_DIR, "neuroLens_architecture.svg")

def create_architecture_diagram():
    # Dimensions: 1920 x 1280 (16:10.66) high resolution canvas
    fig = plt.figure(figsize=(19.2, 12.8), dpi=200, facecolor="#080C14")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#080C14")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Helper function: draw styled card
    def draw_card(x, y, w, h, bg="#111827", border="#1E293B", radius=1.2, lw=1.5):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.2,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        return box

    # Helper function: draw connector arrow
    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", lw=2.0, style="->", rad=0.0):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            connectionstyle=f"arc3,rad={rad}",
            color=color,
            linewidth=lw,
            mutation_scale=16,
            zorder=3
        )
        ax.add_patch(arrow)

    # Helper function: draw pill badge
    def draw_badge(x, y, text, bg="#1E293B", fg="#94A3B8", fontsize=8.5):
        w = len(text) * 0.48 + 1.6
        box = FancyBboxPatch(
            (x - w/2, y - 0.9), w, 1.8,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor=bg, edgecolor="none", zorder=4
        )
        ax.add_patch(box)
        ax.text(x, y, text, color=fg, fontsize=fontsize, fontweight="bold",
                ha="center", va="center", family="sans-serif", zorder=5)

    # =========================================================================
    # 1. HEADER SECTION (Top Banner)
    # =========================================================================
    draw_card(3, 91.5, 94, 6.5, bg="#0E1526", border="#1E293B", radius=1.0, lw=1.2)
    ax.text(5, 95.8, "NEUROLENS", color="#38BDF8", fontsize=19, fontweight="heavy", family="sans-serif", zorder=5)
    ax.text(17.8, 95.8, "· SYSTEM ARCHITECTURE & DATAFLOW", color="#F8FAFC", fontsize=18, fontweight="bold", family="sans-serif", zorder=5)
    ax.text(5, 93.2, "Automated Multi-Class Brain MRI Diagnostic Workspace  |  MobileNetV2 Two-Stage Transfer Learning & Grad-CAM Explainability",
            color="#94A3B8", fontsize=10.5, family="sans-serif", zorder=5)

    draw_badge(88, 94.8, "PRODUCTION RUNTIME", bg="#0369A1", fg="#F0F9FF", fontsize=9)
    draw_badge(75, 94.8, "STREAMLIT 1.36+ / KERAS 3", bg="#1E293B", fg="#38BDF8", fontsize=9)

    # =========================================================================
    # 2. SECTION A: OFFLINE MODEL DEVELOPMENT PIPELINE (Top Panel: Y = 68 to 89.5)
    # =========================================================================
    # Background panel for Section A
    draw_card(3, 68.5, 94, 21.5, bg="#0B1120", border="#25334E", radius=1.5, lw=1.5)
    
    # Section Header Badge
    draw_badge(18, 88.5, "PHASE 1: OFFLINE MODEL DEVELOPMENT & TRAINING (RESEARCH STAGE)", bg="#312E81", fg="#C7D2FE", fontsize=9.5)
    ax.text(95, 88.5, "Independent Training & Leakage-Free Validation (Zero Leakage)", color="#6366F1", fontsize=9, ha="right", va="center", family="sans-serif", zorder=5)

    # Step A1: Raw Dataset
    draw_card(4.5, 71.0, 16.5, 15.0, bg="#111827", border="#4338CA", radius=1.0)
    draw_badge(12.75, 84.5, "1. CURATED DATASET", bg="#312E81", fg="#A5B4FC", fontsize=8.5)
    ax.text(12.75, 81.2, "Kaggle Deduplicated\nBrain MRI Collection", color="#F8FAFC", fontsize=10.5, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)
    ax.text(12.75, 76.5, "• Manifest Total: 6,507 MRI\n• Active Dataset: 6,461 Scans\n• Quarantined Excl: 46 Scans\n• 4 Target Tumor Classes", color="#94A3B8", fontsize=8.5, ha="center", va="center", family="sans-serif", zorder=4)
    draw_badge(12.75, 72.8, "6,507 Manifest Scans", bg="#1E1B4B", fg="#818CF8", fontsize=8)

    draw_arrow(21.3, 78.5, 24.5, 78.5, color="#818CF8", lw=2.2)

    # Step A2: Leakage Audit & Split C
    draw_card(24.5, 71.0, 17.0, 15.0, bg="#111827", border="#4338CA", radius=1.0)
    draw_badge(33.0, 84.5, "2. LEAKAGE AUDIT & SPLIT", bg="#312E81", fg="#A5B4FC", fontsize=8.5)
    ax.text(33.0, 81.2, "417-Group Audit &\nReconciled Clean Split", color="#F8FAFC", fontsize=10.5, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)
    ax.text(33.0, 76.5, "• Training: 4,752 scans (73.6%)\n• Validation: 487 scans (7.5%)\n• Test Set: 1,222 scans (18.9%)\n• Strict Patient Group Isolation", color="#94A3B8", fontsize=8.5, ha="center", va="center", family="sans-serif", zorder=4)
    draw_badge(33.0, 72.8, "Zero Train-Test Leakage", bg="#1E1B4B", fg="#34D399", fontsize=8)

    draw_arrow(41.8, 78.5, 45.0, 78.5, color="#818CF8", lw=2.2)

    # Step A3: Transfer Learning Backbone & Two-Stage Training
    draw_card(45.0, 71.0, 24.5, 15.0, bg="#111827", border="#4338CA", radius=1.0)
    draw_badge(57.25, 84.5, "3. TWO-STAGE TRANSFER LEARNING", bg="#312E81", fg="#A5B4FC", fontsize=8.5)
    ax.text(57.25, 81.2, "MobileNetV2 Deep Architecture", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)
    ax.text(57.25, 76.5, "• Stage 1 Warmup: Frozen Backbone (LR = 1e-3, Adam)\n• Stage 2 Tuning: Unfrozen Top Blocks (LR = 1e-5, Cosine)\n• Depthwise Separable Convolutions & Bottlenecks\n• Inverted Residual Blocks (Feature Map: 7 × 7 × 1280)", color="#94A3B8", fontsize=8.5, ha="center", va="center", family="sans-serif", zorder=4)
    draw_badge(57.25, 72.8, "Pretrained ImageNet Weights", bg="#1E1B4B", fg="#F59E0B", fontsize=8)

    draw_arrow(69.8, 78.5, 73.0, 78.5, color="#818CF8", lw=2.2)

    # Step A4: Final Model Evaluation & Checkpoint Serialization
    draw_card(73.0, 71.0, 22.5, 15.0, bg="#111827", border="#047857", radius=1.0, lw=1.8)
    draw_badge(84.25, 84.5, "4. EVALUATION & ACTIVE ARTIFACT", bg="#064E3B", fg="#6EE7B7", fontsize=8.5)
    ax.text(84.25, 81.2, "Verified Production Checkpoint", color="#10B981", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)
    ax.text(84.25, 76.5, "• Clean Test Accuracy: 86.50% (1,222 unseen scans)\n• Clean Validation Accuracy: 90.35% (487 scans)\n• Checkpoint: mobilenet_two_stage_final_unweighted.keras\n• Cryptographic SHA-256: 4C2A8E08C90948E5...", color="#CBD5E1", fontsize=8.5, ha="center", va="center", family="sans-serif", zorder=4)
    draw_badge(84.25, 72.8, "models/ (24.5 MB Deployed)", bg="#064E3B", fg="#34D399", fontsize=8)

    # Connecting Arrow from Offline Model to Online Ingestion (Downwards Flow)
    draw_arrow(84.25, 70.8, 84.25, 63.8, color="#10B981", lw=2.8, style="-|>", rad=0.0)
    draw_badge(84.25, 66.8, "MODEL LOAD VIA @st.cache_resource", bg="#064E3B", fg="#A7F3D0", fontsize=8.5)

    # =========================================================================
    # 3. SECTION B: REAL-TIME PREDICTION & EXPLAINABILITY (Y = 3.5 to 64.0)
    # =========================================================================
    draw_card(3, 3.5, 94, 60.5, bg="#090E1A", border="#1E3A8A", radius=1.5, lw=1.8)
    
    # Section Header Badge
    draw_badge(20.5, 62.5, "PHASE 2: REAL-TIME INFERENCE & EXPLAINABILITY ENGINE (STREAMLIT WORKSPACE)", bg="#0C4A6E", fg="#BAE6FD", fontsize=9.5)
    ax.text(95, 62.5, "Zero Server State Mutation  ·  Isolated Downstream Grad-CAM  ·  Deterministic Predictions", color="#38BDF8", fontsize=9, ha="right", va="center", family="sans-serif", zorder=5)

    # Column 1: Client Ingestion & Validation (Left: X = 4.5 to 22.0)
    draw_card(4.5, 33.0, 17.5, 26.5, bg="#0F172A", border="#0284C7", radius=1.2)
    draw_badge(13.25, 57.5, "STAGE 1: IMAGE INGESTION", bg="#0369A1", fg="#E0F2FE", fontsize=8.5)
    ax.text(13.25, 53.8, "Client Upload & Inspection", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)
    
    # Subcard: Streamlit File Uploader
    draw_card(5.5, 43.5, 15.5, 8.5, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(13.25, 49.5, "Streamlit File Uploader", color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(13.25, 46.0, "Accepted: .JPG, .JPEG, .PNG\nMax Size: 25 MB", color="#94A3B8", fontsize=8.5, ha="center", va="center", zorder=4)

    # Subcard: Verification & Preprocessing
    draw_card(5.5, 34.5, 15.5, 8.0, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(13.25, 40.0, "Integrity Verification", color="#34D399", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(13.25, 36.8, "• PIL.Image.verify() check\n• Graceful Exception Handler\n• Zero Unhandled Crashes", color="#94A3B8", fontsize=8, ha="center", va="center", zorder=4)

    draw_arrow(22.3, 46.2, 25.5, 46.2, color="#38BDF8", lw=2.2)

    # Column 2: Mathematical Preprocessing & Invariant Scaling (X = 25.5 to 43.5)
    draw_card(25.5, 33.0, 18.0, 26.5, bg="#0F172A", border="#0284C7", radius=1.2)
    draw_badge(34.5, 57.5, "STAGE 2: PREPROCESSING", bg="#0369A1", fg="#E0F2FE", fontsize=8.5)
    ax.text(34.5, 53.8, "Tensor Normalization", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)
    
    draw_card(26.5, 43.5, 16.0, 8.5, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(34.5, 49.5, "Spatial Standardization", color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(34.5, 46.0, "• Convert to 3-Channel RGB\n• Resize: 224 × 224 px (Bilinear)", color="#94A3B8", fontsize=8.5, ha="center", va="center", zorder=4)

    draw_card(26.5, 34.5, 16.0, 8.0, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(34.5, 40.0, "MobileNetV2 Normalization", color="#F59E0B", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(34.5, 36.8, "• mobilenet_v2.preprocess_input\n• Numerical Scale: [-1.0, +1.0]\n• Input Shape: (1, 224, 224, 3)", color="#94A3B8", fontsize=8, ha="center", va="center", zorder=4)

    draw_arrow(43.8, 46.2, 47.0, 46.2, color="#38BDF8", lw=2.2)

    # Column 3: Model Inference & Softmax Distribution (Center-Right: X = 47.0 to 69.5)
    draw_card(47.0, 33.0, 22.5, 26.5, bg="#0F172A", border="#059669", radius=1.2, lw=1.6)
    draw_badge(58.25, 57.5, "STAGE 3: INFERENCE ENGINE", bg="#065F46", fg="#A7F3D0", fontsize=8.5)
    ax.text(58.25, 53.8, "Forward Classification Pass", color="#10B981", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)

    draw_card(48.0, 43.5, 20.5, 8.5, bg="#1E293B", border="#047857", radius=0.8)
    ax.text(58.25, 49.5, "Trained MobileNetV2 Checkpoint", color="#34D399", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(58.25, 46.0, "raw_pred = model.predict(tensor)\nCached Instance | Latency ~45ms (CPU)", color="#CBD5E1", fontsize=8.5, ha="center", va="center", family="monospace", zorder=4)

    draw_card(48.0, 34.5, 20.5, 8.0, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(58.25, 40.2, "Softmax Categorical Output", color="#F8FAFC", fontsize=9, fontweight="bold", ha="center", va="center", zorder=4)
    
    # 4 Category tags
    draw_badge(51.0, 37.0, "Glioma", bg="#7F1D1D", fg="#FCA5A5", fontsize=7.5)
    draw_badge(55.8, 37.0, "Meningioma", bg="#78350F", fg="#FDE68A", fontsize=7.5)
    draw_badge(61.2, 37.0, "No Tumor", bg="#064E3B", fg="#6EE7B7", fontsize=7.5)
    draw_badge(66.2, 37.0, "Pituitary", bg="#581C87", fg="#E9D5FF", fontsize=7.5)

    # Split Arrows from Stage 3:
    # 1. Straight right to UI Presentation
    draw_arrow(69.8, 46.2, 73.0, 46.2, color="#34D399", lw=2.4)
    # 2. Downward to Grad-CAM Explainer
    draw_arrow(58.25, 32.8, 58.25, 24.5, color="#F43F5E", lw=2.4, style="-|>")
    draw_badge(58.25, 28.5, "PREDICTED CLASS (c) & FEATURE MAP", bg="#881337", fg="#FFE4E6", fontsize=8)

    # Column 4: Top Results Presentation (Upper Right: X = 73.0 to 95.5, Y = 33.0 to 59.5)
    draw_card(73.0, 33.0, 22.5, 26.5, bg="#0F172A", border="#0284C7", radius=1.2)
    draw_badge(84.25, 57.5, "STAGE 4: DIAGNOSTIC DASHBOARD", bg="#0369A1", fg="#E0F2FE", fontsize=8.5)
    ax.text(84.25, 53.8, "Result Presentation & UI", color="#F8FAFC", fontsize=11, fontweight="bold", ha="center", va="center", family="sans-serif", zorder=4)

    draw_card(74.0, 44.0, 20.5, 8.0, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(84.25, 49.5, "Diagnostic Result Card", color="#38BDF8", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(84.25, 46.5, "Top-1 Category + Softmax Likelihood %\nNormalized Distribution Progress Bars", color="#94A3B8", fontsize=8.5, ha="center", va="center", zorder=4)

    draw_card(74.0, 34.5, 20.5, 8.5, bg="#1E293B", border="#334155", radius=0.8)
    ax.text(84.25, 40.2, "Plotly Distribution Chart", color="#A78BFA", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(84.25, 36.8, "Interactive Donut Visualization\nClinical Advisory & Metadata Grid", color="#94A3B8", fontsize=8.5, ha="center", va="center", zorder=4)

    # =========================================================================
    # Bottom Row: EXPLAINABLE AI & REPORT GENERATION (Y = 5.0 to 24.0)
    # =========================================================================

    # Stage 5: Explainable AI Engine (Bottom Left/Center: X = 4.5 to 58.5)
    draw_card(4.5, 5.0, 54.0, 19.5, bg="#180D1E", border="#BE185D", radius=1.2, lw=1.8)
    draw_badge(20.0, 22.8, "STAGE 5: EXPLAINABLE AI ENGINE (GRAD-CAM)", bg="#831843", fg="#FCE7F3", fontsize=9)
    ax.text(56.5, 22.8, "scripts/gradcam_explainer.py  ·  Zero Classifier Modification", color="#F43F5E", fontsize=8.5, ha="right", va="center", family="sans-serif", zorder=4)

    # Sub-Box 1: Target Layer Tap
    draw_card(5.5, 7.0, 16.5, 13.5, bg="#27112E", border="#9D174D", radius=0.8)
    ax.text(13.75, 17.5, "1. Feature Map Extraction", color="#FB7185", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(13.75, 13.0, "• Target Layer: out_relu\n• Tensor Shape: (1, 7, 7, 1280)\n• Terminal MobileNetV2 block\n• High semantic receptive field", color="#FBCFE8", fontsize=8.2, ha="center", va="center", zorder=4)
    draw_badge(13.75, 8.8, "Layer: out_relu", bg="#831843", fg="#FFF1F2", fontsize=7.5)

    draw_arrow(22.3, 13.5, 24.2, 13.5, color="#F43F5E", lw=2.0)

    # Sub-Box 2: Gradient Computation
    draw_card(24.2, 7.0, 16.5, 13.5, bg="#27112E", border="#9D174D", radius=0.8)
    ax.text(32.45, 17.5, "2. Gradient Backprop", color="#FB7185", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(32.45, 13.0, "• tf.GradientTape w.r.t yc\n• GAP: alpha_k = (1/Z) ΣΣ d(yc)/dA\n• Channel importance weighting\n• WeakKeyDictionary Cache (<0.05ms)", color="#FBCFE8", fontsize=8.2, ha="center", va="center", zorder=4)
    draw_badge(32.45, 8.8, "ReLU Activation Filter", bg="#831843", fg="#FFF1F2", fontsize=7.5)

    draw_arrow(41.0, 13.5, 42.9, 13.5, color="#F43F5E", lw=2.0)

    # Sub-Box 3: Bilinear Heatmap Overlay
    draw_card(42.9, 7.0, 14.5, 13.5, bg="#27112E", border="#9D174D", radius=0.8)
    ax.text(50.15, 17.5, "3. Heatmap Composite", color="#FB7185", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(50.15, 13.0, "• 7 × 7 -> Native Res (Bilinear)\n• Colormap: 'jet' (RGB)\n• Alpha blend: α = 0.4\n• Dual scan side-by-side UI", color="#FBCFE8", fontsize=8.2, ha="center", va="center", zorder=4)
    draw_badge(50.15, 8.8, "Non-Diagnostic Notice", bg="#831843", fg="#FFF1F2", fontsize=7.5)

    # Arrow from Grad-CAM to Report Engine
    draw_arrow(58.8, 14.5, 62.0, 14.5, color="#F43F5E", lw=2.2)

    # Stage 6: Multi-Format Report Generation (Bottom Right: X = 62.0 to 95.5)
    draw_card(62.0, 5.0, 33.5, 19.5, bg="#131B2E", border="#7C3AED", radius=1.2, lw=1.8)
    draw_badge(78.75, 22.8, "STAGE 6: MULTI-FORMAT CLINICAL REPORT GENERATION", bg="#5B21B6", fg="#EDE9FE", fontsize=9)

    # Download TXT Card
    draw_card(63.2, 7.0, 15.0, 13.5, bg="#1E1B4B", border="#6D28D9", radius=0.8)
    ax.text(70.7, 17.5, "Plain-Text Report (.txt)", color="#C4B5FD", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(70.7, 13.0, "• Authoritative ASCII summary\n• Primary classification + score\n• Per-class probability breakdown\n• Input specs & execution status\n• Research & clinical disclaimer", color="#DDD6FE", fontsize=8.2, ha="center", va="center", zorder=4)
    draw_badge(70.7, 8.8, "Brain_Tumor_Report.txt", bg="#4C1D95", fg="#F5F3FF", fontsize=7.5)

    # Download PDF Card
    draw_card(79.2, 7.0, 15.0, 13.5, bg="#1E1B4B", border="#6D28D9", radius=0.8)
    ax.text(86.7, 17.5, "Vector PDF Report (.pdf)", color="#C4B5FD", fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=4)
    ax.text(86.7, 13.0, "• Matplotlib Vector PDF canvas\n• High-DPI professional layout\n• Visual color-coded card\n• Technical parameters table\n• Formal radiologist disclaimer", color="#DDD6FE", fontsize=8.2, ha="center", va="center", zorder=4)
    draw_badge(86.7, 8.8, "Brain_Tumor_Report.pdf", bg="#4C1D95", fg="#F5F3FF", fontsize=7.5)

    # Connecting Arrow from Stage 4 (UI Dashboard) to Reports
    draw_arrow(84.25, 32.8, 84.25, 24.8, color="#7C3AED", lw=2.2, style="-|>")
    draw_badge(84.25, 28.5, "EXPORT ENGINE DISPATCH", bg="#4C1D95", fg="#DDD6FE", fontsize=8)

    # Save outputs
    print(f"Saving high-resolution PNG architecture diagram to: {PNG_PATH}")
    fig.savefig(PNG_PATH, format="png", dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f"Saving vector SVG architecture diagram to: {SVG_PATH}")
    fig.savefig(SVG_PATH, format="svg", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("Architecture diagram generation successful!")

if __name__ == "__main__":
    create_architecture_diagram()
