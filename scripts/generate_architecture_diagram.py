"""Script to generate publication-grade System Architecture Diagram for Chapter 3.2.

Generates:
1. diagrams/system_architecture.png (300 DPI, publication quality)
2. diagrams/system_architecture.svg (Scalable vector format)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, PathPatch
from matplotlib.path import Path as MplPath

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS_DIR = PROJECT_ROOT / "diagrams"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)


def create_system_architecture_diagram():
    # Setup Figure with high-res canvas (19.0 inches x 12.0 inches)
    fig, ax = plt.subplots(figsize=(19.0, 12.0), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.axis("off")

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    FONT_FAMILY = "DejaVu Sans"
    COLOR_PRIMARY_DARK = "#0F172A"      # Slate 900
    COLOR_TEXT_DARK = "#1E293B"         # Slate 800
    COLOR_TEXT_MUTED = "#475569"        # Slate 600

    def draw_card(x, y, w, h, title, subtitle="", edge_color="#2563EB", bg_color="#FFFFFF", header_bg="#2563EB"):
        # Card Body
        card = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=1.0",
            facecolor=bg_color,
            edgecolor=edge_color,
            linewidth=1.6,
            zorder=2
        )
        ax.add_patch(card)

        # Header bar
        header_h = 2.8
        h_rect = FancyBboxPatch(
            (x, y + h - header_h), w, header_h,
            boxstyle="round,pad=0.0,rounding_size=1.0",
            facecolor=header_bg,
            edgecolor=edge_color,
            linewidth=1.0,
            zorder=3
        )
        ax.add_patch(h_rect)
        ax.add_patch(Rectangle((x, y + h - header_h), w, 1.0, facecolor=header_bg, edgecolor="none", zorder=4))

        # Title text
        ax.text(
            x + w / 2, y + h - header_h / 2,
            title,
            ha="center", va="center",
            fontsize=9.5, fontweight="bold",
            color="#FFFFFF",
            family=FONT_FAMILY,
            zorder=5
        )

        # Subtitle box
        if subtitle:
            sub_rect = Rectangle((x + 0.3, y + h - header_h - 1.4), w - 0.6, 1.3, facecolor="#F1F5F9", edgecolor="none", zorder=4)
            ax.add_patch(sub_rect)
            ax.text(
                x + w / 2, y + h - header_h - 0.75,
                subtitle,
                ha="center", va="center",
                fontsize=7.3, fontstyle="italic",
                color="#334155",
                family=FONT_FAMILY,
                zorder=5
            )

        return card

    def draw_straight_arrow(x1, y1, x2, y2, text="", color="#2563EB", lw=1.8, ls="-", label_offset=(0, 0.75), font_size=7.2):
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>,head_length=6.0,head_width=3.8",
            color=color,
            linewidth=lw,
            linestyle=ls,
            zorder=6
        )
        ax.add_patch(arrow)
        if text:
            mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
            ax.text(
                mx, my, text,
                ha="center", va="center",
                fontsize=font_size, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor=color, linewidth=0.7, alpha=0.95),
                zorder=7
            )

    def draw_orthogonal_arrow(points, text="", color="#2563EB", lw=1.6, ls="-", label_pos=None, font_size=7.5):
        verts = points
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
        path = MplPath(verts, codes)
        patch = PathPatch(path, facecolor="none", edgecolor=color, linewidth=lw, linestyle=ls, zorder=6)
        ax.add_patch(patch)

        p_penult = points[-2]
        p_last = points[-1]
        arrow_head = FancyArrowPatch(
            p_penult, p_last,
            connectionstyle="arc3,rad=0.0",
            arrowstyle="-|>,head_length=6.0,head_width=3.8",
            color=color,
            linewidth=lw,
            linestyle=ls,
            zorder=7
        )
        ax.add_patch(arrow_head)

        if text and label_pos:
            ax.text(
                label_pos[0], label_pos[1], text,
                ha="center", va="center",
                fontsize=font_size, fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#FFFFFF", edgecolor=color, linewidth=0.8, alpha=0.96),
                zorder=8
            )

    # 1. HEADER SECTION
    ax.text(
        50, 67.8,
        "BRAIN TUMOR MRI CLASSIFICATION: END-TO-END SYSTEM ARCHITECTURE",
        ha="center", va="center",
        fontsize=15.5, fontweight="bold",
        color=COLOR_PRIMARY_DARK,
        family=FONT_FAMILY
    )
    ax.text(
        50, 65.8,
        "The Apollo University | School of Technology | Dept. of CSE (AI & DS) | Chapter 3.2 Technical Specification",
        ha="center", va="center",
        fontsize=9.8, fontweight="normal",
        color=COLOR_TEXT_MUTED,
        family=FONT_FAMILY
    )

    # Track 1 Background (Offline Training & Benchmarking)
    ax.add_patch(FancyBboxPatch(
        (1.5, 35.8), 97.0, 28.5,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        facecolor="#F0F9FF", edgecolor="#BAE6FD", linewidth=1.4, linestyle="--", zorder=1
    ))
    ax.text(
        3.5, 63.0, "PIPELINE 1: OFFLINE DATASET AUDIT, MODEL TRAINING & BENCHMARK EVALUATION",
        fontsize=9.2, fontweight="bold", color="#0369A1", family=FONT_FAMILY, zorder=2
    )

    # Track 2 Background (Online Interactive Inference & Decision Support)
    ax.add_patch(FancyBboxPatch(
        (1.5, 2.2), 97.0, 29.8,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        facecolor="#FAF5FF", edgecolor="#E9D5FF", linewidth=1.4, linestyle="--", zorder=1
    ))
    ax.text(
        3.5, 30.6, "PIPELINE 2: ONLINE INTERACTIVE INFERENCE, EXPLAINABILITY (Grad-CAM) & CLINICAL DECISION SUPPORT",
        fontsize=9.2, fontweight="bold", color="#6B21A8", family=FONT_FAMILY, zorder=2
    )

    # =========================================================================
    # PIPELINE 1: OFFLINE WORKFLOW CARDS
    # Total width available: 94.0. Gaps: 3 x 2.8 = 8.4. Remaining: 85.6 / 4 = ~21.4
    # y: 38.0, h: 23.5
    # =========================================================================

    # Card 1.1: Dataset Repository
    c11_x, c11_w = 3.5, 21.5
    draw_card(
        c11_x, 38.0, c11_w, 23.5,
        "1.1 Multiclass MRI Dataset",
        "dataset/ (7,200 total images)",
        edge_color="#0284C7", header_bg="#0284C7"
    )
    dataset_bullets = [
        ("• 4 Balanced Diagnostic Classes:", True),
        ("   - Glioma: 1,400 Train | 400 Test", False),
        ("   - Meningioma: 1,400 Train | 400 Test", False),
        ("   - No Tumor: 1,400 Train | 400 Test", False),
        ("   - Pituitary: 1,400 Train | 400 Test", False),
        ("• Physical Directory Organization:", True),
        ("   - dataset/Training: 5,600 images (77.8%)", False),
        ("   - dataset/Testing: 1,600 images (22.2%)", False),
        ("• Data Integrity & Leakage Audit:", True),
        ("   - 0 SHA-256 exact cross-split duplicates", False),
        ("   - 26,589 near-duplicate adjacent axial", False),
        ("     slices audited (Kaggle benchmark)", False),
    ]
    for i, (line, is_head) in enumerate(dataset_bullets):
        ax.text(
            c11_x + 0.8, 56.4 - i * 1.35, line,
            fontsize=7.6, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 1.2: Data Loading & Augmentation
    c12_x, c12_w = 27.8, 20.8
    draw_card(
        c12_x, 38.0, c12_w, 23.5,
        "1.2 Loading & Augmentation",
        "src/data/dataset.py & augmentation.py",
        edge_color="#0D9488", header_bg="#0D9488"
    )
    aug_bullets = [
        ("• Stratified Split (80/20 on Training):", True),
        ("   - Train Split: 4,480 images (augmented)", False),
        ("   - Val Split: 1,120 images (unaugmented)", False),
        ("• Conservative Medical Augmentation:", True),
        ("   - Rotation: +/- 5% (~18 deg)", False),
        ("   - Zoom: +/- 8% (height & width)", False),
        ("   - Translation: +/- 5% spatial shift", False),
        ("   - Brightness & Contrast: +/- 10%", False),
        ("   - Horizontal Flip: Enabled (Bilateral)", False),
        ("   - Vertical Flip: Prohibited (Inverts axes)", False),
        ("• Standard Preprocessing: Resize 224x224", True),
    ]
    for i, (line, is_head) in enumerate(aug_bullets):
        ax.text(
            c12_x + 0.8, 56.4 - i * 1.35, line,
            fontsize=7.6, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 1.3: Deep Learning Training
    c13_x, c13_w = 51.4, 22.8
    draw_card(
        c13_x, 38.0, c13_w, 23.5,
        "1.3 Offline Model Training",
        "scripts/train_cnn.py & train_efficientnet.py",
        edge_color="#2563EB", header_bg="#2563EB"
    )
    training_bullets = [
        ("• Custom CNN Baseline (train_cnn.py):", True),
        ("   - 4 Conv-Pool blocks (32, 64, 128, 256)", False),
        ("   - GAP -> Dense(128) -> Dense(4, softmax)", False),
        ("   - Total Params: 421,828 (100% trainable)", False),
        ("• EfficientNet-B0 (train_efficientnet.py):", True),
        ("   - ImageNet Pretrained MBConv Backbone", False),
        ("   - Stage 1: Frozen backbone (lr=1e-3)", False),
        ("   - Stage 2: Fine-tune top 20 layers (lr=1e-5)", False),
        ("   - Trainable Head Params: 164,484", False),
        ("• Model Checkpoint Artifacts:", True),
        ("   - baseline_cnn.keras, efficientnet_b0.keras", False),
    ]
    for i, (line, is_head) in enumerate(training_bullets):
        ax.text(
            c13_x + 0.8, 56.4 - i * 1.35, line,
            fontsize=7.6, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 1.4: Benchmark Evaluation & Metrics
    c14_x, c14_w = 77.0, 21.5
    draw_card(
        c14_x, 38.0, c14_w, 23.5,
        "1.4 Benchmark Evaluation",
        "scripts/evaluate.py & display_metrics.py",
        edge_color="#4338CA", header_bg="#4338CA"
    )
    eval_bullets = [
        ("• Untouched Test Set Evaluation (N=1,600):", True),
        ("   - 400 Glioma, 400 Meningioma,", False),
        ("     400 No Tumor, 400 Pituitary", False),
        ("• Verified Benchmark Performance:", True),
        ("   - Custom CNN: 68.75% Acc | 0.6624 F1", False),
        ("     (Glioma Recall Bottleneck: 41.25%)", False),
        ("   - EfficientNet-B0: 87.50% Acc | 0.8699 F1", False),
        ("     (Macro ROC-AUC: 0.9718 | 95% Bootstrap CI)", False),
        ("• Machine-Readable Outputs (reports/):", True),
        ("   - final_metrics.json, classification_report.csv", False),
        ("   - Confusion Matrices & ROC Curves", False),
    ]
    for i, (line, is_head) in enumerate(eval_bullets):
        ax.text(
            c14_x + 0.8, 56.4 - i * 1.35, line,
            fontsize=7.6, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Pipeline 1 Horizontal Arrows
    draw_straight_arrow(c11_x + c11_w, 49.5, c12_x, 49.5, "5.6k Train", color="#0284C7", lw=1.6, label_offset=(0, 0.75), font_size=7.0)
    draw_straight_arrow(c12_x + c12_w, 49.5, c13_x, 49.5, "4.5k Aug", color="#0D9488", lw=1.6, label_offset=(0, 0.75), font_size=7.0)
    draw_straight_arrow(c13_x + c13_w, 49.5, c14_x, 49.5, "Models", color="#2563EB", lw=1.6, label_offset=(0, 0.75), font_size=7.0)

    # Pipeline 1 Orthogonal Route for Test Set (Clean channel at y=36.7)
    draw_orthogonal_arrow(
        [(c11_x + c11_w, 39.5), (26.4, 39.5), (26.4, 36.7), (75.6, 36.7), (75.6, 39.5), (c14_x, 39.5)],
        text="Held-Out Benchmark Split: 1,600 Isolated Test Images (dataset/Testing/)",
        color="#4338CA", lw=1.6, ls="--", label_pos=(51.0, 36.7), font_size=7.4
    )

    # =========================================================================
    # PIPELINE 2: ONLINE INTERACTIVE INFERENCE CARDS
    # Total width: 94.0. Gaps: 4 x 2.2 = 8.8. Remaining: 85.2 / 5 = ~17.0
    # y: 5.0, h: 24.2
    # =========================================================================

    # Card 2.1: Image Input & Validation
    c21_x, c21_w = 3.5, 16.5
    draw_card(
        c21_x, 5.0, c21_w, 24.2,
        "2.1 Input Ingestion",
        "Streamlit File Uploader / Test Samples",
        edge_color="#7E22CE", header_bg="#7E22CE"
    )
    input_bullets = [
        ("• Input Modalities:", True),
        ("   - Drag-and-Drop Upload (JPG, PNG)", False),
        ("   - Preloaded Test Set Samples", False),
        ("• Validation & Guardrails:", True),
        ("   - PIL Format Integrity Check", False),
        ("   - 3-Channel RGB Casting", False),
        ("   - Corrupted File Rejection", False),
        ("   - Empty Input State Guard", False),
        ("• Dimension Invariance:", True),
        ("   - Stores original resolution", False),
        ("     for authentic overlay blend", False),
    ]
    for i, (line, is_head) in enumerate(input_bullets):
        ax.text(
            c21_x + 0.8, 23.6 - i * 1.55, line,
            fontsize=7.5, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 2.2: Inference Preprocessing
    c22_x, c22_w = 22.2, 17.0
    draw_card(
        c22_x, 5.0, c22_w, 24.2,
        "2.2 Preprocessing",
        "src/data/preprocessing.py",
        edge_color="#9333EA", header_bg="#9333EA"
    )
    prep_bullets = [
        ("• Standardized Resizing:", True),
        ("   - Anti-Aliased Bilinear Resize", False),
        ("   - Target Resolution: 224 x 224 x 3", False),
        ("• Architecture-Specific Scaling:", True),
        ("   - CNN Baseline: [0, 1] (/ 255.0)", False),
        ("   - EfficientNet-B0: preprocess_input", False),
        ("     (Rescales to [-1, 1] range)", False),
        ("• Batch Tensor Generation:", True),
        ("   - Formats input (1, 224, 224, 3)", False),
        ("   - Keeps unscaled uint8 RGB image", False),
    ]
    for i, (line, is_head) in enumerate(prep_bullets):
        ax.text(
            c22_x + 0.8, 23.6 - i * 1.55, line,
            fontsize=7.5, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 2.3: Checkpoint Inference Engine
    c23_x, c23_w = 41.4, 17.5
    draw_card(
        c23_x, 5.0, c23_w, 24.2,
        "2.3 Checkpoint Inference",
        "tf.keras.models.load_model",
        edge_color="#2563EB", header_bg="#2563EB"
    )
    infer_bullets = [
        ("• Model Checkpoint Loading:", True),
        ("   - efficientnet_b0.keras (Primary)", False),
        ("   - baseline_cnn.keras (Baseline)", False),
        ("   - Cached via @st.cache_resource", False),
        ("   - ZERO retraining on inference", False),
        ("• Feedforward Pass:", True),
        ("   - Feature extraction through", False),
        ("     inverted residual MBConv stages", False),
        ("   - Global Average Pooling (GAP)", False),
        ("   - Raw Output: Softmax logits (1, 4)", False),
    ]
    for i, (line, is_head) in enumerate(infer_bullets):
        ax.text(
            c23_x + 0.8, 23.6 - i * 1.55, line,
            fontsize=7.5, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 2.4: Calibration & Uncertainty Gating
    c24_x, c24_w = 61.1, 17.0
    draw_card(
        c24_x, 5.0, c24_w, 24.2,
        "2.4 Calibration & Gating",
        "src/evaluation/calibration.py",
        edge_color="#D97706", header_bg="#D97706"
    )
    calib_bullets = [
        ("• Decoupled Temperature Scaling:", True),
        ("   - EfficientNet: T = 1.0993 (fitted", False),
        ("     strictly on validation split)", False),
        ("   - Custom CNN: T = 1.0000 (Raw)", False),
        ("• Predictive Shannon Entropy:", True),
        ("   - H(p) = -SUM(p_i * ln(p_i))", False),
        ("• Tri-State Clinical Gating:", True),
        ("   - High Confidence (Green >= 85%)", False),
        ("   - Moderate Confidence (Yellow 60-85%)", False),
        ("   - Clinical Expert Flag (Red < 60%)", False),
    ]
    for i, (line, is_head) in enumerate(calib_bullets):
        ax.text(
            c24_x + 0.8, 23.6 - i * 1.55, line,
            fontsize=7.5, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Card 2.5: Explainability Engine & Streamlit UI
    c25_x, c25_w = 80.3, 17.2
    draw_card(
        c25_x, 5.0, c25_w, 24.2,
        "2.5 Explainability & UI",
        "app.py & src/explainability/",
        edge_color="#BE123C", header_bg="#BE123C"
    )
    ui_bullets = [
        ("• Visual Saliency Attribution:", True),
        ("   - Target: top_activation (7x7x1280)", False),
        ("   - Grad-CAM (1st order gradients)", False),
        ("   - Grad-CAM++ (Higher order partials)", False),
        ("   - JET colormap, alpha blend (0.40)", False),
        ("• Interactive Streamlit Dashboard:", True),
        ("   - 3-Panel Saliency Visualization", False),
        ("   - Calibrated Probability Bar Chart", False),
        ("   - Clinical Uncertainty Guidance", False),
        ("• Mandatory Medical Disclaimer", True),
    ]
    for i, (line, is_head) in enumerate(ui_bullets):
        ax.text(
            c25_x + 0.8, 23.6 - i * 1.55, line,
            fontsize=7.5, fontweight="bold" if is_head else "normal",
            color=COLOR_TEXT_DARK if is_head else COLOR_TEXT_MUTED,
            family=FONT_FAMILY, zorder=5
        )

    # Pipeline 2 Horizontal Straight Arrows
    draw_straight_arrow(c21_x + c21_w, 16.0, c22_x, 16.0, "Raw MRI", color="#7E22CE", lw=1.6, label_offset=(0, 0.75), font_size=7.0)
    draw_straight_arrow(c22_x + c22_w, 16.0, c23_x, 16.0, "(1,224,224,3)", color="#9333EA", lw=1.6, label_offset=(0, 0.75), font_size=7.0)
    draw_straight_arrow(c23_x + c23_w, 16.0, c24_x, 16.0, "Raw Probs", color="#2563EB", lw=1.6, label_offset=(0, 0.75), font_size=7.0)
    draw_straight_arrow(c24_x + c24_w, 16.0, c25_x, 16.0, "Calibrated", color="#D97706", lw=1.6, label_offset=(0, 0.75), font_size=7.0)

    # Cross-Pipeline Clean Routing:
    # 1. Model Checkpoint Loading (Card 1.3 to Card 2.3)
    # Channel between tracks is y in [32.0, 35.8]. Line runs at y=33.8
    draw_orthogonal_arrow(
        [(c13_x + c13_w / 2, 38.0), (c13_x + c13_w / 2, 33.8), (c23_x + c23_w / 2, 33.8), (c23_x + c23_w / 2, 29.2)],
        text="Loads Saved Checkpoints (baseline_cnn.keras / efficientnet_b0.keras)",
        color="#1D4ED8", lw=1.6, ls="-", label_pos=(56.0, 33.8), font_size=7.4
    )

    # 2. Evaluation Metrics Ingestion (Card 1.4 to Card 2.5)
    # Drops straight down from Card 1.4 into Card 2.5
    draw_straight_arrow(
        c14_x + c14_w / 2, 38.0, c25_x + c25_w / 2, 29.2,
        text="Injects reports/final_metrics.json",
        color="#4338CA", lw=1.6, ls=":", label_offset=(0.0, 1.8), font_size=7.3
    )

    # 3. Clean Grad-CAM Feature Map & Gradients Bus (Underneath cards 2.3 and 2.4 at y=3.5)
    draw_orthogonal_arrow(
        [(c23_x + c23_w / 2, 5.0), (c23_x + c23_w / 2, 3.5), (c25_x + c25_w / 2, 3.5), (c25_x + c25_w / 2, 5.0)],
        text="Target Conv Feature Maps (top_activation: 7x7x1280) & Target Class Gradients dy/dA",
        color="#BE123C", lw=1.6, ls="-.", label_pos=(68.5, 3.5), font_size=7.3
    )

    # Bottom Safety & Academic Integrity Banner
    safety_rect = FancyBboxPatch(
        (3.5, 0.5), 93.0, 1.2,
        boxstyle="round,pad=0.0,rounding_size=0.4",
        facecolor="#FEF2F2", edgecolor="#FCA5A5", linewidth=1.0, zorder=8
    )
    ax.add_patch(safety_rect)
    ax.text(
        50, 1.1,
        "ACADEMIC & CLINICAL NOTICE: Educational decision-support prototype. Grad-CAM attribution reflects model feature weighting, NOT semantic tumor segmentation or biological margins.",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color="#991B1B", family=FONT_FAMILY, zorder=9
    )

    png_path = DIAGRAMS_DIR / "system_architecture.png"
    svg_path = DIAGRAMS_DIR / "system_architecture.svg"

    print(f"Saving high-resolution PNG (300 DPI) to {png_path} ...")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")

    print(f"Saving scalable vector graphic (SVG) to {svg_path} ...")
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")

    plt.close()
    print("Architecture diagram generation completed successfully!")


if __name__ == "__main__":
    create_system_architecture_diagram()
