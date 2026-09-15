"""Streamlit application for Brain Tumor MRI Classification & Grad-CAM Explainability.

Educational interface using:
- Pretrained EfficientNetB0 transfer learning model (efficientnet_b0.keras)
- Grad-CAM heatmap visualization from final convolutional layer (top_activation)
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from gradcam import (
    DEFAULT_MODEL_PATH,
    TARGET_CONV_LAYER,
    compute_gradcam,
    load_and_prepare_model,
    overlay_heatmap,
    preprocess_image,
)
from preprocess import CLASS_NAMES


DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

SAMPLE_IMAGES = {
    "Select a sample image...": None,
    "Glioma Sample (Testing/glioma)": "dataset/Testing/glioma/Te-gl_10.jpg",
    "Meningioma Sample (Testing/meningioma)": "dataset/Testing/meningioma/Te-aug-me_1.jpg",
    "No Tumor Sample (Testing/notumor)": "dataset/Testing/notumor/Te-no_1.jpg",
    "Pituitary Sample (Testing/pituitary)": "dataset/Testing/pituitary/Te-pi_1.jpg",
}


@st.cache_resource(show_spinner="Loading EfficientNetB0 model checkpoint...")
def get_cached_model():
    """Load and cache the EfficientNetB0 model and Grad-CAM feature extractor."""
    full_model, feature_extractor, head_layers = load_and_prepare_model(DEFAULT_MODEL_PATH)
    return full_model, feature_extractor, head_layers


def main() -> None:
    st.set_page_config(
        page_title="Brain MRI Classification & Explainability",
        page_icon=".",
        layout="wide",
    )

    st.title("Brain MRI Classification & Explainability")
    st.markdown("#### EfficientNetB0 + Grad-CAM")
    st.markdown(
        "Upload a brain MRI scan to classify it into one of four categories "
        "(**Glioma**, **Meningioma**, **No Tumor**, or **Pituitary**) and visualize "
        "the model's region of interest using Grad-CAM."
    )
    st.divider()

    # Load cached model
    try:
        full_model, feature_extractor, head_layers = get_cached_model()
    except Exception as exc:
        st.error(f"Error loading model from `{DEFAULT_MODEL_PATH}`: {exc}")
        st.stop()

    # Layout: Input section
    col_input_left, col_input_right = st.columns([2, 1])

    with col_input_left:
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["jpg", "jpeg", "png"],
            help="Select an MRI image in JPG or PNG format.",
        )

    with col_input_right:
        selected_sample_label = st.selectbox(
            "Or test with a sample MRI",
            options=list(SAMPLE_IMAGES.keys()),
            help="Choose a pre-loaded sample MRI from the project dataset.",
        )

    # Determine input image
    image_to_process: Image.Image | None = None
    image_source_desc = ""

    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            image_source_desc = f"Uploaded file: `{uploaded_file.name}`"
        except Exception as exc:
            st.error(f"Failed to read uploaded image: {exc}. Please upload a valid image.")
            st.stop()
    elif selected_sample_label and SAMPLE_IMAGES[selected_sample_label]:
        sample_path = Path(SAMPLE_IMAGES[selected_sample_label])
        if sample_path.is_file():
            image_to_process = Image.open(sample_path).convert("RGB")
            image_source_desc = f"Sample: `{sample_path.name}`"
        else:
            st.warning(f"Sample file not found at: {sample_path}")

    if image_to_process is None:
        st.info("👆 Please upload a brain MRI scan or choose a sample image above to proceed.")
        return

    st.write(f"**Selected Image:** {image_source_desc} (Original Dimensions: {image_to_process.size[0]} × {image_to_process.size[1]} px)")

    # Display image preview and Analyze button
    col_preview, col_action = st.columns([1, 2])
    with col_preview:
        st.image(image_to_process, caption="Selected Brain MRI", width=260)
    with col_action:
        st.write("")
        st.write("")
        analyze_btn = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    if not analyze_btn:
        st.caption("Click **Analyze Image** to run classification and generate Grad-CAM visualization.")
        return

    # Run Analysis
    with st.spinner("Processing MRI and generating Grad-CAM explainability..."):
        try:
            preprocessed_tensor, orig_image_rgb, orig_size = preprocess_image(image_to_process)
        except Exception as exc:
            st.error(f"Preprocessing error: {exc}")
            return

        # 1. Model Prediction
        try:
            raw_predictions = full_model.predict(preprocessed_tensor, verbose=0)[0]
            pred_idx = int(np.argmax(raw_predictions))
            pred_raw_class = CLASS_NAMES[pred_idx]
            pred_display_name = DISPLAY_NAMES.get(pred_raw_class, pred_raw_class.title())
            confidence = float(raw_predictions[pred_idx])
        except Exception as exc:
            st.error(f"Model inference failed: {exc}")
            return

        # 2. Grad-CAM Calculation
        gradcam_success = False
        heatmap_rgb, overlay_rgb = None, None
        try:
            heatmap_2d = compute_gradcam(
                feature_extractor=feature_extractor,
                head_layers=head_layers,
                preprocessed_image=preprocessed_tensor,
                target_class_idx=pred_idx,
            )
            heatmap_rgb, overlay_rgb = overlay_heatmap(
                heatmap_2d=heatmap_2d,
                orig_image_rgb=orig_image_rgb,
                alpha=0.40,
                colormap_name="jet",
            )
            gradcam_success = True
        except Exception as exc:
            st.warning(f"Grad-CAM generation encountered an issue: {exc}")

    # Display Results
    st.divider()
    st.subheader("Classification Results")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.metric(
            label="Predicted Tumor Type",
            value=pred_display_name,
        )

    with res_col2:
        st.metric(
            label="Confidence",
            value=f"{confidence * 100:.2f}%",
        )

    # Class Probability Chart
    st.write("##### Class Probability Distribution")
    prob_data = {
        "Class": [DISPLAY_NAMES.get(c, c.title()) for c in CLASS_NAMES],
        "Probability (%)": [float(raw_predictions[i] * 100) for i in range(len(CLASS_NAMES))],
    }
    df_probs = pd.DataFrame(prob_data).set_index("Class")
    st.bar_chart(df_probs, horizontal=True)

    # Progress bars for detailed breakdown
    for i, c in enumerate(CLASS_NAMES):
        p_val = float(raw_predictions[i])
        c_label = DISPLAY_NAMES.get(c, c.title())
        st.write(f"**{c_label}**: {p_val * 100:.2f}%")
        st.progress(min(max(p_val, 0.0), 1.0))

    # Grad-CAM Visualization Section
    st.divider()
    st.subheader("Grad-CAM Explainability")

    if gradcam_success and heatmap_rgb is not None and overlay_rgb is not None:
        g_col1, g_col2, g_col3 = st.columns(3)

        with g_col1:
            st.image(orig_image_rgb, caption="Original MRI", width="stretch")
        with g_col2:
            st.image(
                heatmap_rgb,
                caption=f"Grad-CAM Heatmap ({TARGET_CONV_LAYER})",
                width="stretch",
            )
        with g_col3:
            st.image(
                overlay_rgb,
                caption=f"Grad-CAM Overlay ({pred_display_name})",
                width="stretch",
            )

        st.info(
            "The highlighted regions show image features that influenced the model's prediction. "
            "This visualization is for educational/research purposes and does not constitute medical diagnosis."
        )
    else:
        st.warning("Grad-CAM overlay could not be rendered for this image.")


if __name__ == "__main__":
    main()
