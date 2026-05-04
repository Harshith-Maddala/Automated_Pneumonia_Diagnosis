import io
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


st.set_page_config(
    page_title="CheXpert Uncertainty Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


DISEASE_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
]

UNCERTAINTY_CONFIG = {
    "U-Zeros": {
        "results_dir": "evaluation_outputs_uzeros",
        "models": {
            "ResNet-50": "resnet50",
            "DenseNet-121": "densenet121",
            "CaMCheX": "camchex_dual_view",
            "MultiView Dual Branch": "multiview_dual_branch",
        },
        "densenet_checkpoint": "best_densenet121_uzeros.pth",
        "camchex_checkpoint": "best_camchex_uzeros.pth",
    },
    "U-Ones": {
        "results_dir": "evaluation_outputs_uones",
        "models": {
            "ResNet-50": "resnet50_uones",
            "DenseNet-121": "densenet121_uones",
            "CaMCheX": "camchex_dual_view_uones",
            "MultiView Dual Branch": "multiview_dual_branch_uones",
        },
        "densenet_checkpoint": "best_densenet121_uones.pth",
        "camchex_checkpoint": "best_camchex_uones.pth",
    },
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_single_view_transform():
    return transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_camchex_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CaMCheXDualView(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        frontal_backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        lateral_backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        frontal_in_features = frontal_backbone.classifier[2].in_features
        lateral_in_features = lateral_backbone.classifier[2].in_features
        frontal_backbone.classifier = nn.Identity()
        lateral_backbone.classifier = nn.Identity()
        self.frontal_encoder = frontal_backbone
        self.lateral_encoder = lateral_backbone
        fusion_dim = frontal_in_features + lateral_in_features
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, frontal_img, lateral_img):
        frontal_feat = torch.flatten(self.frontal_encoder(frontal_img), 1)
        lateral_feat = torch.flatten(self.lateral_encoder(lateral_img), 1)
        return self.classifier(torch.cat([frontal_feat, lateral_feat], dim=1))


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_single(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(output).detach().cpu().numpy()[0]

    def generate_dual(self, frontal_tensor, lateral_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(frontal_tensor, lateral_tensor)
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=frontal_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, torch.sigmoid(output).detach().cpu().numpy()[0]


def cam_to_bbox(cam, threshold=0.5):
    binary = (cam > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


def densenet_gradcam_visualize(image, model, disease_cols, device,
                                confidence_threshold=0.3, cam_threshold=0.5):
    transform = get_single_view_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    target_layer = model.features.denseblock4.denselayer16.conv2
    gradcam = GradCAM(model, target_layer)
    model.eval()

    detected = []
    with torch.enable_grad():
        for i, disease in enumerate(disease_cols):
            cam, probs = gradcam.generate_single(input_tensor, class_idx=i)
            confidence = probs[i]
            if confidence >= confidence_threshold:
                bbox = cam_to_bbox(cam, threshold=cam_threshold)
                detected.append({
                    "disease": disease,
                    "confidence": float(confidence),
                    "bbox": bbox,
                })

    orig_w, orig_h = image.size
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(detected), 1)))
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap="gray")
    for det, color in zip(detected, colors):
        bbox = det["bbox"]
        color_rgb = color[:3]
        if bbox is not None:
            scale_x = orig_w / 320
            scale_y = orig_h / 320
            x, y, w, h = bbox
            x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                           edgecolor=color_rgb, facecolor="none"))
            ax.text(x, y-5, f"{det['disease']}  {det['confidence']:.2f}",
                    color="white", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor=color_rgb, alpha=0.7, pad=2, edgecolor="none"))
    ax.axis("off")
    ax.set_title("DenseNet-121 — Disease Detection with Confidence Scores", fontsize=12)
    plt.tight_layout()
    return fig, detected


def camchex_gradcam_visualize(frontal_image, lateral_image, model, disease_cols,
                               device, confidence_threshold=0.3, cam_threshold=0.5):
    transform = get_camchex_transform()
    frontal_tensor = transform(frontal_image).unsqueeze(0).to(device)
    lateral_tensor = transform(lateral_image).unsqueeze(0).to(device)
    frontal_tensor.requires_grad_(True)

    # Last Conv2d in last CNBlock of deepest ConvNeXt stage
    target_layer = model.frontal_encoder.features[7][2].block[0]
    gradcam = GradCAM(model, target_layer)
    model.eval()

    detected = []
    with torch.enable_grad():
        for i, disease in enumerate(disease_cols):
            cam, probs = gradcam.generate_dual(frontal_tensor, lateral_tensor, class_idx=i)
            confidence = probs[i]
            if confidence >= confidence_threshold:
                bbox = cam_to_bbox(cam, threshold=cam_threshold)
                detected.append({
                    "disease": disease,
                    "confidence": float(confidence),
                    "bbox": bbox,
                })

    orig_w, orig_h = frontal_image.size
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(detected), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(frontal_image, cmap="gray")
    for det, color in zip(detected, colors):
        bbox = det["bbox"]
        color_rgb = color[:3]
        if bbox is not None:
            scale_x = orig_w / 224
            scale_y = orig_h / 224
            x, y, w, h = bbox
            x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
            axes[0].add_patch(patches.Rectangle((x, y), w, h, linewidth=2,
                                                edgecolor=color_rgb, facecolor="none"))
            axes[0].text(x, y-5, f"{det['disease']}  {det['confidence']:.2f}",
                         color="white", fontsize=8, fontweight="bold",
                         bbox=dict(facecolor=color_rgb, alpha=0.7, pad=2, edgecolor="none"))
    axes[0].axis("off")
    axes[0].set_title("Frontal — GradCAM Detection", fontsize=11)

    axes[1].imshow(lateral_image, cmap="gray")
    axes[1].axis("off")
    axes[1].set_title("Lateral — Reference View", fontsize=11)

    plt.suptitle("CaMCheX Dual-View Detection with Confidence Scores", fontsize=13)
    plt.tight_layout()
    return fig, detected


@st.cache_resource(show_spinner=False)
def build_densenet_model(checkpoint_path: str):
    device = get_device()
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, len(DISEASE_COLS))
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def build_camchex_model(checkpoint_path: str):
    device = get_device()
    model = CaMCheXDualView(len(DISEASE_COLS))
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_uploaded_image(uploaded_file):
    return Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")


def confidence_label(v):
    if v >= 0.9: return "Very High"
    if v >= 0.75: return "High"
    if v >= 0.6: return "Moderate"
    return "Low"


def build_prediction_table(probabilities, calibration_df=None):
    probs = probabilities.flatten()
    preds = (probs >= 0.5).astype(int)
    confidences = np.maximum(probs, 1 - probs)
    rows = []
    for i, disease in enumerate(DISEASE_COLS):
        row = {
            "Disease": disease,
            "Predicted Label": "Positive" if preds[i] == 1 else "Negative",
            "Probability": float(probs[i]),
            "Confidence Estimate": float(confidences[i]),
            "Confidence Level": confidence_label(float(confidences[i])),
        }
        if calibration_df is not None and disease in set(calibration_df["Disease"]):
            calib_row = calibration_df[calibration_df["Disease"] == disease].iloc[0]
            row["Validation AUROC"] = calib_row.get("AUROC", np.nan)
            row["Validation AUPRC"] = calib_row.get("AUPRC", np.nan)
            row["Validation ECE"] = calib_row.get("ECE", np.nan)
            row["Validation Brier"] = calib_row.get("Brier Score", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_single_image(model, image):
    device = get_device()
    tensor = get_single_view_transform()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor)).cpu().numpy()
    return probs


def predict_dual_image(model, frontal_image, lateral_image):
    device = get_device()
    ft = get_camchex_transform()(frontal_image).unsqueeze(0).to(device)
    lt = get_camchex_transform()(lateral_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(ft, lt)).cpu().numpy()
    return probs


def show_detected_table(detected):
    if detected:
        st.markdown("#### Detected Diseases")
        df = pd.DataFrame([
            {
                "Disease": d["disease"],
                "Confidence": round(d["confidence"], 4),
                "Confidence Level": confidence_label(d["confidence"]),
                "Prediction": "Positive",
            }
            for d in sorted(detected, key=lambda x: x["confidence"], reverse=True)
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No diseases detected above threshold. Try lowering the confidence threshold.")


def get_results_root(uncertainty_name, custom_root):
    if custom_root:
        return Path(custom_root)
    return Path(UNCERTAINTY_CONFIG[uncertainty_name]["results_dir"])


def get_model_results_dir(uncertainty_name, model_name, custom_root):
    root = get_results_root(uncertainty_name, custom_root)
    model_slug = UNCERTAINTY_CONFIG[uncertainty_name]["models"][model_name]
    return root / model_slug


def load_model_metrics(uncertainty_name, model_name, custom_root=""):
    model_dir = get_model_results_dir(uncertainty_name, model_name, custom_root)
    per_disease_path = model_dir / "per_disease_metrics.csv"
    overall_path = model_dir / "overall_metrics.csv"
    per_disease_df = pd.read_csv(per_disease_path) if per_disease_path.exists() else None
    overall_df = pd.read_csv(overall_path) if overall_path.exists() else None
    return model_dir, per_disease_df, overall_df


def load_comparison_table(custom_roots):
    rows = []
    for uncertainty_name in UNCERTAINTY_CONFIG:
        for model_name, model_slug in UNCERTAINTY_CONFIG[uncertainty_name]["models"].items():
            root = get_results_root(uncertainty_name, custom_roots.get(uncertainty_name, ""))
            overall_path = root / model_slug / "overall_metrics.csv"
            if overall_path.exists():
                overall = pd.read_csv(overall_path).iloc[0].to_dict()
                overall["Uncertainty Strategy"] = uncertainty_name
                overall["Model"] = model_name
                rows.append(overall)
    if not rows:
        return None
    comparison_df = pd.DataFrame(rows)
    cols = [
        "Uncertainty Strategy", "Model",
        "macro_auroc", "micro_auroc", "macro_auprc", "micro_auprc",
        "micro_ece", "micro_brier_score", "macro_f1", "micro_f1",
    ]
    existing_cols = [col for col in cols if col in comparison_df.columns]
    return comparison_df[existing_cols].sort_values(
        by=["Uncertainty Strategy", "Model"]
    ).reset_index(drop=True)


def show_plot_if_available(image_path, caption):
    if image_path.exists():
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        st.info(f"{caption} not found at `{image_path}`.")


def render_project_page(custom_roots):
    st.title("Thoracic Disease Detection Under Uncertainty")
    st.markdown("""
        This dashboard presents our chest X-ray classification project built on the CheXpert-style
        multi-label setting, where the goal is to detect multiple thoracic diseases and findings
        from radiographic images while handling label uncertainty carefully. We explored two
        uncertainty strategies, `U-Zeros` and `U-Ones`, and evaluated them across four deep learning
        models spanning single-view and dual-view architectures.

        The system compares `ResNet-50`, `DenseNet-121`, `CaMCheX`, and a `MultiView Dual Branch`
        network using classification performance, ranking performance, and calibration quality. This
        allows us to study not only whether the models predict the right diseases, but also whether
        their confidence scores are trustworthy enough for a clinical decision-support setting.
    """)

    st.subheader("Strategy And Model Comparison")
    comparison_df = load_comparison_table(custom_roots)
    if comparison_df is None:
        st.warning("No evaluation result folders found. Configure paths in the sidebar.")
    else:
        st.dataframe(comparison_df, use_container_width=True)

    left, right = st.columns(2)
    with left:
        selected_uncertainty = st.selectbox(
            "Select Uncertainty Strategy", options=list(UNCERTAINTY_CONFIG.keys())
        )
    with right:
        selected_model = st.selectbox(
            "Select Deep Learning Model",
            options=list(UNCERTAINTY_CONFIG[selected_uncertainty]["models"].keys()),
        )

    model_dir, per_disease_df, overall_df = load_model_metrics(
        selected_uncertainty, selected_model, custom_roots.get(selected_uncertainty, "")
    )

    st.subheader(f"Detailed Results: {selected_uncertainty} | {selected_model}")

    if overall_df is None or per_disease_df is None:
        st.error(f"Could not find saved evaluation CSVs in `{model_dir}`.")
        return

    metric_cols = st.columns(6)
    r = overall_df.iloc[0]
    metric_cols[0].metric("Macro AUROC", f"{r.get('macro_auroc', np.nan):.4f}")
    metric_cols[1].metric("Micro AUROC", f"{r.get('micro_auroc', np.nan):.4f}")
    metric_cols[2].metric("Macro AUPRC", f"{r.get('macro_auprc', np.nan):.4f}")
    metric_cols[3].metric("Micro AUPRC", f"{r.get('micro_auprc', np.nan):.4f}")
    metric_cols[4].metric("Micro ECE",   f"{r.get('micro_ece', np.nan):.4f}")
    metric_cols[5].metric("Brier Score", f"{r.get('micro_brier_score', np.nan):.4f}")

    st.markdown("#### Per-Disease Metrics")
    st.dataframe(per_disease_df, use_container_width=True)

    plot_left, plot_right = st.columns(2)
    with plot_left:
        show_plot_if_available(model_dir / "roc_curves.png", "AUROC Curves")
        show_plot_if_available(model_dir / "reliability_diagram.png", "Reliability Diagram")
    with plot_right:
        show_plot_if_available(model_dir / "pr_curves.png", "AUPRC Curves")
        show_plot_if_available(model_dir / "confidence_histogram.png", "Confidence Histogram")


def render_prediction_page(checkpoint_dir, custom_roots):
    st.title("X-Ray Inference And Confidence Estimation")
    st.markdown("""
        Upload a frontal X-ray for **DenseNet-121** to get GradCAM bounding boxes with confidence
        scores. Upload frontal + lateral X-rays for **CaMCheX** to get dual-view GradCAM
        predictions — bounding boxes are shown on the frontal image, lateral shown for reference.
    """)

    uncertainty_name = "U-Zeros"
    st.info("Inference is fixed to the U-Zeros versions of DenseNet-121 and CaMCheX.")

    st.markdown("#### Detection Settings")
    col1, col2 = st.columns(2)
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold", min_value=0.1, max_value=0.9,
            value=0.3, step=0.05,
            help="Only show diseases the model is at least this confident about",
        )
    with col2:
        cam_threshold = st.slider(
            "GradCAM Threshold", min_value=0.3, max_value=0.8,
            value=0.5, step=0.05,
            help="Controls how tight the bounding boxes are around detected regions",
        )

    st.markdown("---")
    dense_col, cam_col = st.columns(2)

    # ── DenseNet-121 + GradCAM ────────────────────────────────
    with dense_col:
        st.subheader("DenseNet-121 + GradCAM")
        st.caption("Single frontal view — GradCAM on last DenseBlock layer")

        dense_upload = st.file_uploader(
            "Upload Frontal X-ray",
            type=["png", "jpg", "jpeg"],
            key="dense_upload",
        )
        dense_ckpt = Path(checkpoint_dir) / UNCERTAINTY_CONFIG[uncertainty_name]["densenet_checkpoint"]
        _, dense_calib_df, _ = load_model_metrics(
            uncertainty_name, "DenseNet-121", custom_roots.get(uncertainty_name, "")
        )

        if dense_upload is not None:
            image = load_uploaded_image(dense_upload)
            try:
                with st.spinner("Running DenseNet-121 GradCAM inference..."):
                    model = build_densenet_model(str(dense_ckpt))
                    fig, detected = densenet_gradcam_visualize(
                        image, model, DISEASE_COLS, get_device(),
                        confidence_threshold=confidence_threshold,
                        cam_threshold=cam_threshold,
                    )
                st.pyplot(fig)
                plt.close(fig)
                show_detected_table(detected)
                with st.expander("Full prediction table — all 13 diseases"):
                    probs = predict_single_image(model, image)
                    st.dataframe(build_prediction_table(probs, dense_calib_df), use_container_width=True)
            except FileNotFoundError as exc:
                st.error(str(exc))

    # ── CaMCheX + GradCAM ────────────────────────────────────
    with cam_col:
        st.subheader("CaMCheX + GradCAM")
        st.caption("Frontal + lateral views — GradCAM on frontal ConvNeXt features")

        cam_front_upload = st.file_uploader(
            "Upload Frontal X-ray",
            type=["png", "jpg", "jpeg"],
            key="cam_front_upload",
        )
        cam_lateral_upload = st.file_uploader(
            "Upload Lateral X-ray",
            type=["png", "jpg", "jpeg"],
            key="cam_lateral_upload",
        )
        cam_ckpt = Path(checkpoint_dir) / UNCERTAINTY_CONFIG[uncertainty_name]["camchex_checkpoint"]
        _, cam_calib_df, _ = load_model_metrics(
            uncertainty_name, "CaMCheX", custom_roots.get(uncertainty_name, "")
        )

        if cam_front_upload is not None and cam_lateral_upload is not None:
            frontal_image = load_uploaded_image(cam_front_upload)
            lateral_image = load_uploaded_image(cam_lateral_upload)
            try:
                with st.spinner("Running CaMCheX GradCAM inference..."):
                    model = build_camchex_model(str(cam_ckpt))
                    fig, detected = camchex_gradcam_visualize(
                        frontal_image, lateral_image, model, DISEASE_COLS, get_device(),
                        confidence_threshold=confidence_threshold,
                        cam_threshold=cam_threshold,
                    )
                st.pyplot(fig)
                plt.close(fig)
                show_detected_table(detected)
                with st.expander("Full prediction table — all 13 diseases"):
                    probs = predict_dual_image(model, frontal_image, lateral_image)
                    st.dataframe(build_prediction_table(probs, cam_calib_df), use_container_width=True)
            except FileNotFoundError as exc:
                st.error(str(exc))

        elif cam_front_upload is not None and cam_lateral_upload is None:
            st.warning("Please also upload a lateral X-ray for CaMCheX.")
        elif cam_lateral_upload is not None and cam_front_upload is None:
            st.warning("Please also upload a frontal X-ray for CaMCheX.")


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Dashboard Controls")
    page = st.radio("Navigate", options=["Project Overview", "Model Inference"])

    st.markdown("#### Data Paths")
    checkpoint_dir = st.text_input(
        "Checkpoint Directory", value=".",
        help="Folder containing the saved .pth model files.",
    )
    uzero_results_override = st.text_input(
        "U-Zeros Results Directory", value="",
        help="Optional override for U-Zeros evaluation outputs.",
    )
    uones_results_override = st.text_input(
        "U-Ones Results Directory", value="",
        help="Optional override for U-Ones evaluation outputs.",
    )
    st.markdown("---")
    st.markdown("#### About")
    st.markdown(
        "Deep Learning Based Detection of Thorax Diseases from "
        "Chest X-rays with Confidence Estimation."
    )
    st.markdown("**Team:** Maddala Harshith, Tadapaneni Snehitha")
    st.markdown("**Institution:** George Washington University")

custom_roots = {
    "U-Zeros": uzero_results_override,
    "U-Ones": uones_results_override,
}

if page == "Project Overview":
    render_project_page(custom_roots)
else:
    render_prediction_page(checkpoint_dir, custom_roots)