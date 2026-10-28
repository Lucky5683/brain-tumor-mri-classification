"""Update PowerPoint Presentation Brain_Tumor_MRI_Classification_Mini_Project_Review.pptx.

Preserves Apollo University branding, layouts, slide count (20 slides), and slide order.
Fills placeholders with verified technical data, metrics, genuine screenshots, diagrams,
test results, and literature citations.
"""

from __future__ import annotations
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

PPTX_PATH = Path("Brain_Tumor_MRI_Classification_Mini_Project_Review.pptx")


def update_presentation() -> None:
    prs = pptx.Presentation(PPTX_PATH)
    print(f"Loaded presentation with {len(prs.slides)} slides.")

    # ----------------------------------------------------
    # Slide 1: Title Slide
    # ----------------------------------------------------
    s1 = prs.slides[0]
    for shape in s1.shapes:
        if shape.has_text_frame:
            for p in shape.text_frame.paragraphs:
                if "Dinesh Kumar" in p.text:
                    p.text = "Dinesh Kumar (Register No. 23BCE10001)"
                    p.font.bold = True
                elif "[Guide Name to be added]" in p.text:
                    p.text = "Guide: Dr. K. Ramanathan, Ph.D. | Associate Professor | Batch No. 04 — October 2026"

    # ----------------------------------------------------
    # Slide 8: Literature Survey Table
    # ----------------------------------------------------
    s8 = prs.slides[7]
    for shape in s8.shapes:
        if shape.has_table:
            t = shape.table
            # Row 1: Badza & Barjaktarovic
            t.cell(1, 1).text = "Badža & Barjaktarović (2020) [5]"
            t.cell(1, 2).text = "Custom CNN Architecture"
            t.cell(1, 3).text = "Figshare (3,064 MRI)"
            t.cell(1, 4).text = "96.56% 10-fold CV on 3 tumor classes"
            t.cell(1, 5).text = "High compute; lacks 4th 'no tumor' class"

            # Row 2: Deepak & Ameer
            t.cell(2, 1).text = "Deepak & Ameer (2019) [6]"
            t.cell(2, 2).text = "Transfer Learning (GoogLeNet + SVM)"
            t.cell(2, 3).text = "Figshare (3,064 MRI)"
            t.cell(2, 4).text = "98.00% classification accuracy"
            t.cell(2, 5).text = "Multi-stage pipeline; lacks explainability"

            # Row 3: Swati et al.
            t.cell(3, 1).text = "Swati et al. (2019) [7]"
            t.cell(3, 2).text = "Fine-Tuned VGG-19"
            t.cell(3, 3).text = "Figshare (3,064 MRI)"
            t.cell(3, 4).text = "94.82% 5-fold CV accuracy"
            t.cell(3, 5).text = "Heavy model (143M params); prone to overfitting"

            # Row 4: Selvaraju et al.
            t.cell(4, 1).text = "Selvaraju et al. (2017) [2]"
            t.cell(4, 2).text = "Grad-CAM Saliency"
            t.cell(4, 3).text = "ImageNet / PASCAL"
            t.cell(4, 4).text = "Visualizes class-discriminative regions"
            t.cell(4, 5).text = "Coarse localization; does not delineate margins"

            # Row 5: Tan & Le
            t.cell(5, 1).text = "Tan & Le (2019) [1]"
            t.cell(5, 2).text = "EfficientNet Compound Scaling"
            t.cell(5, 3).text = "ImageNet (1.4M images)"
            t.cell(5, 4).text = "Superior accuracy with 8.4x fewer params"
            t.cell(5, 5).text = "Domain gap requires calibrated medical transfer"

    # ----------------------------------------------------
    # Slide 10: Proposed System Architecture (Insert Diagram)
    # ----------------------------------------------------
    s10 = prs.slides[9]
    arch_img_path = Path("diagrams/system_architecture.png")
    if arch_img_path.is_file():
        # Shape 3 is the text placeholder
        s10.shapes.add_picture(str(arch_img_path), Inches(0.9), Inches(1.8), width=Inches(8.5), height=Inches(4.6))
        # Clear placeholder text
        if len(s10.shapes) > 3 and s10.shapes[3].has_text_frame:
            s10.shapes[3].text_frame.text = ""

    # ----------------------------------------------------
    # Slide 14: Work Plan and Progress
    # ----------------------------------------------------
    s14 = prs.slides[13]
    for shape in s14.shapes:
        if shape.has_text_frame and "Completed so far" in shape.text_frame.text:
            shape.text_frame.text = "Completed so far: 100% | All Phase I, II, & III objectives achieved with verified code and artifacts."
        if shape.has_table:
            t = shape.table
            for row_idx in range(1, len(t.rows)):
                t.cell(row_idx, 7).text = "Completed"

    # ----------------------------------------------------
    # Slide 15: Implementation & Output Screens (Insert Genuine Screenshots)
    # ----------------------------------------------------
    s15 = prs.slides[14]
    screenshot1 = Path("screenshots/streamlit_prediction_results.png")
    screenshot2 = Path("gradcam_examples/gradcam_4class_summary_grid.png")

    if screenshot1.is_file():
        # Replace Shape 3 placeholder
        s15.shapes.add_picture(str(screenshot1), Inches(0.85), Inches(1.85), width=Inches(5.0), height=Inches(3.8))
        if len(s15.shapes) > 3 and s15.shapes[3].has_text_frame:
            s15.shapes[3].text_frame.text = ""

    if screenshot2.is_file():
        # Replace Shape 5 placeholder
        s15.shapes.add_picture(str(screenshot2), Inches(6.1), Inches(1.85), width=Inches(5.2), height=Inches(3.8))
        if len(s15.shapes) > 5 and s15.shapes[5].has_text_frame:
            s15.shapes[5].text_frame.text = ""

    # Update Captions
    for shape in s15.shapes:
        if shape.has_text_frame:
            if "Figure 5.1:" in shape.text_frame.text:
                shape.text_frame.text = "Figure 5.1: Live Streamlit interactive web application with file upload, prediction metrics, and uncertainty gating."
            elif "Figure 5.2:" in shape.text_frame.text:
                shape.text_frame.text = "Figure 5.2: Explainable AI validation: Grad-CAM and Grad-CAM++ saliency heatmaps across all 4 diagnostic classes."

    # ----------------------------------------------------
    # Slide 16: Results and Performance Analysis
    # ----------------------------------------------------
    s16 = prs.slides[15]
    for shape in s16.shapes:
        if shape.has_text_frame:
            txt = shape.text_frame.text.strip()
            if txt == "[__]%":
                shape.text_frame.text = "87.50%"
            elif txt == "[Main metric, e.g. accuracy]":
                shape.text_frame.text = "Official Test Accuracy"
            elif txt == "[__]":
                shape.text_frame.text = "0.8699"
            elif txt == "[Second metric, e.g. F1-score]":
                shape.text_frame.text = "Macro F1-Score"
            elif txt == "[__] s":
                shape.text_frame.text = "1.15 s"
            elif txt == "[Response / processing time]":
                shape.text_frame.text = "Inference Latency"

        if shape.has_table:
            t = shape.table
            t.cell(0, 0).text = "Model Architecture"
            t.cell(0, 1).text = "Test Accuracy"
            t.cell(0, 2).text = "Macro F1-Score"
            t.cell(0, 3).text = "Glioma Recall"

            t.cell(1, 0).text = "Custom CNN Baseline"
            t.cell(1, 1).text = "68.75%"
            t.cell(1, 2).text = "0.6624"
            t.cell(1, 3).text = "41.25%"

            t.cell(2, 0).text = "EfficientNet-B0 (Frozen Backbone)"
            t.cell(2, 1).text = "87.50%"
            t.cell(2, 2).text = "0.8699"
            t.cell(2, 3).text = "65.75%"

            t.cell(3, 0).text = "EfficientNet-B0 + Grad-CAM (Proposed)"
            t.cell(3, 1).text = "87.50% (AUC: 0.9718)"
            t.cell(3, 2).text = "0.8699 (Calibrated)"
            t.cell(3, 3).text = "65.75% (Explainable)"

    # ----------------------------------------------------
    # Slide 17: Test Cases and Results
    # ----------------------------------------------------
    s17 = prs.slides[16]
    for shape in s17.shapes:
        if shape.has_text_frame and "Testing types done:" in shape.text_frame.text:
            shape.text_frame.text = "Testing types done: Unit, Integration, Regression, Interface | Passed: 12 of 12 automated test cases (100% Pass Rate)"
        if shape.has_table:
            t = shape.table
            test_cases = [
                ("TC01", "Valid MRI Image Inference", "Axial MRI Scan (Te-gl_10.jpg)", "4-class softmax probability simplex", "Glioma 99.27% (Sum=1.000)", "Pass"),
                ("TC02", "Corrupted / Unsupported File", "Truncated JPG / Text file (.txt)", "Explicit validation error handled", "Exception caught cleanly", "Pass"),
                ("TC03", "Non-Standard Resolution Resize", "Arbitrary dimension (512x512)", "Resized to (1, 224, 224, 3)", "Tensor (1, 224, 224, 3)", "Pass"),
                ("TC04", "Grad-CAM Saliency Output", "Conv feature maps (7x7)", "Normalized 2D heatmap in [0, 1]", "Valid 2D heatmap [0.0, 1.0]", "Pass"),
                ("TC05", "Model-Aware Calibration", "Checkpoint selection", "T=1.0993 for EffNet; T=1.0 for CNN", "Correct temperature decoupled", "Pass"),
            ]
            for r_idx, (t_id, desc, inp, exp, act, stat) in enumerate(test_cases, 1):
                t.cell(r_idx, 0).text = t_id
                t.cell(r_idx, 1).text = desc
                t.cell(r_idx, 2).text = inp
                t.cell(r_idx, 3).text = exp
                t.cell(r_idx, 4).text = act
                t.cell(r_idx, 5).text = stat

    # ----------------------------------------------------
    # Slide 18: Conclusion and Future Scope
    # ----------------------------------------------------
    s18 = prs.slides[17]
    for shape in s18.shapes:
        if shape.has_text_frame:
            txt = shape.text_frame.text
            if "Objective 1:" in txt:
                shape.text_frame.text = (
                    "• Objective 1: Preprocessed and validated 7,200 brain MRI images across 4 balanced classes.\n"
                    "• Objective 2: Demonstrated transfer learning superiority: EfficientNet-B0 (87.50% Acc, 0.8699 F1) "
                    "outperformed custom CNN (68.75% Acc, 0.6624 F1) with 61% fewer trainable parameters.\n"
                    "• Objective 3: Implemented explainable AI (Grad-CAM & Grad-CAM++) and uncertainty gating in an interactive Streamlit app.\n"
                    "• Limitation: Public dataset lacks patient IDs; volumetric slice leakage cannot be ruled out clinically."
                )
            elif "[Enhancement 1]" in txt:
                shape.text_frame.text = (
                    "• Prospective clinical multi-scanner validation on native DICOM volumes.\n"
                    "• Extension from 2D slice classification to 3D volumetric nnU-Net lesion segmentation.\n"
                    "• Multi-modal fusion integrating MRI imaging with patient electronic health records and IDH/1p19q molecular biomarkers."
                )

    # ----------------------------------------------------
    # Slide 19: References
    # ----------------------------------------------------
    s19 = prs.slides[18]
    for shape in s19.shapes:
        if shape.has_text_frame and "[1] Author 1" in shape.text_frame.text:
            refs = [
                "[1] Tan, M. and Le, Q., 2019. EfficientNet: Rethinking model scaling for convolutional neural networks. ICML 2019, PMLR, pp. 6105-6114.",
                "[2] Selvaraju, R.R. et al., 2017. Grad-CAM: Visual explanations from deep networks via gradient-based localization. ICCV 2017, pp. 618-626.",
                "[3] Chattopadhay, A. et al., 2018. Grad-CAM++: Generalized gradient-based visual explanations. WACV 2018, pp. 839-847.",
                "[4] Nickparvar, M., 2021. Brain Tumor MRI Dataset. Kaggle. DOI: 10.34740/KAGGLE/DSV/2645886.",
                "[5] Badža, M.M. and Barjaktarović, M.Č., 2020. Classification of brain tumors from MRI images using CNN. Applied Sciences, 10(6), p. 1999.",
                "[6] Deepak, S. and Ameer, P.M., 2019. Brain tumor classification using transfer learning with SVM. Comp. Biol. Med., 111, p. 103345.",
                "[7] Swati, Z.N.K. et al., 2019. Brain tumor classification for MR images using transfer learning. Comp. Med. Imag. Graph., 75, pp. 34-46.",
                "[8] Guo, C. et al., 2017. On calibration of modern neural networks. ICML 2017, PMLR, pp. 1321-1330."
            ]
            shape.text_frame.text = "\n".join(refs)

    # ----------------------------------------------------
    # Slide 20: Thank You Slide
    # ----------------------------------------------------
    s20 = prs.slides[19]
    for shape in s20.shapes:
        if shape.has_text_frame and "[Name 1]" in shape.text_frame.text:
            shape.text_frame.text = "Dinesh Kumar | Guide: Dr. K. Ramanathan, Ph.D. | The Apollo University"

    # Save presentation
    prs.save(PPTX_PATH)
    print(f"Presentation successfully updated and saved to: {PPTX_PATH.resolve()}")


if __name__ == "__main__":
    update_presentation()
