import io
import os
from typing import Tuple
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
from PIL import ImageOps
import streamlit as st
import pathlib, json

# Try TensorFlow/Keras first; fall back cleanly if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Face‑Mask Detector", page_icon="😷", layout="centered")
st.title("😷 Face‑Mask Detector")
st.caption("Subí una imagen y el modelo predecirá si la persona lleva tapabocas.")

# -----------------------------
# Configuration (edit to match your model)
# -----------------------------
# Path to your saved model. Supports a Keras .h5 file or a SavedModel directory
dirpath = "export"

# If your model expects a fixed input size, you can set it here; otherwise we infer it from the model
FALLBACK_TARGET_SIZE: Tuple[int, int] = (224, 224)

# -----------------------------
# Utilities
# -----------------------------


@st.cache_data(show_spinner=False)
def get_model_input_size(model) -> Tuple[int, int]:
    """Infer input spatial size (H, W) from Keras model; fall back to default if unknown."""
    try:
        shape = model.input_shape  # e.g., (None, H, W, C)
        if isinstance(shape, list):
            shape = shape[0]
        h, w = None, None
        if len(shape) == 4:
            _, h, w, _ = shape
        if (h is None) or (w is None):
            return FALLBACK_TARGET_SIZE
        return int(h), int(w)
    except Exception:
        return FALLBACK_TARGET_SIZE
    
@st.cache_resource(show_spinner=False)
def load_model_for_inference(dirpath="export"):
    dirpath = pathlib.Path(dirpath)
    mdl = tf.keras.models.load_model(dirpath / "model.keras", compile=False, custom_objects={"preprocess_input": preprocess_input})
    with open(dirpath / "label_map.json", "r", encoding="utf-8") as f:
        idx2label = {int(k): v for k, v in json.load(f).items()}
    with open(dirpath / "config.json") as f:
        cfg = json.load(f)
    img_size = tuple(cfg.get("img_size", [224, 224]))
    return mdl, idx2label, img_size

def predict_image(pil_img, model, idx2label, img_size=(224, 224)):
    """
    pil_img: objeto PIL.Image, por ejemplo:
        img_bytes = uploaded.read()
        pil_img = Image.open(io.BytesIO(img_bytes))
    """
    # Corregir orientación y asegurar RGB
    pil_img = ImageOps.exif_transpose(pil_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # A tensor -> resize -> float32 (sin escalar; el modelo ya tiene preprocess_input adentro)
    arr = np.array(pil_img)                          # uint8 [0..255]
    img = tf.convert_to_tensor(arr)                  # (H,W,C)
    img = tf.image.resize(img, img_size)             # (224,224,3)
    img = tf.cast(img, tf.float32)

    # Predicción
    logits = model(tf.expand_dims(img, 0), training=False).numpy()[0]
    i = int(np.argmax(logits))
    return {"label": idx2label[i], "index": i, "probs": logits.tolist()}


def predict(model, arr: np.ndarray):
    """Handle binary (sigmoid) or multiclass (softmax) outputs."""
    # Use model.predict to be framework-agnostic within Keras
    preds = model.predict(arr, verbose=0)
    preds = np.array(preds)

    # If output shape is (1, 1) -> binary sigmoid
    if preds.ndim == 2 and preds.shape[1] == 1:
        prob_no_mask = float(preds[0, 0])  # assume 1 -> class index 1 (No Mask)
        # Map to two-class distribution [Mask, No Mask]
        probs = np.array([1.0 - prob_no_mask, prob_no_mask], dtype=np.float32)
        idx = int(probs.argmax())
        return idx, float(probs[idx]), probs

    # If output is (1, C) -> softmax multiclass
    if preds.ndim == 2:
        idx = int(np.argmax(preds[0]))
        prob = float(preds[0, idx])
        return idx, prob, preds[0]

    raise ValueError(f"Formato de salida del modelo no soportado: shape={preds.shape}")


# -----------------------------
# Load model
# -----------------------------
with st.spinner("Cargando modelo…"):
    var =1
    try:
       mdl, idx2label, img_size = load_model_for_inference()
    except Exception as e:
        st.error(f"No se pudo cargar el modelo desde '{pathlib.Path(dirpath)}'. Detalle: {e}")
        st.stop()

st.success(f"Modelo cargado ✅  · Tamaño esperado: {img_size[0]}×{img_size[1]}")

# -----------------------------
# File uploader
# -----------------------------
uploaded = st.file_uploader("Elegí una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"]) 

if uploaded is not None:
    try:
        img_bytes = uploaded.read()
        img = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        st.error(f"No se pudo abrir la imagen: {e}")
        st.stop()

    try:
        out = predict_image(img, mdl, idx2label, img_size)
        print(out)
    except Exception as e:
        st.error(f"Error al predecir: {e}")
        st.stop()
    st.subheader("Resultado")
    st.markdown(f"**Etiqueta predicha:** {out['label']}")

    st.image(img, caption="Imagen cargada", use_container_width=True)

else:
    st.info("Esperando una imagen…")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")