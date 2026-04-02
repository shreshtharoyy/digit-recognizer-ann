import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")
st.caption("Draw a digit (0–9) and click Predict")

# ──────────────────────────────────────────────────────────────
# LOAD MODEL  (cached — loads only once per session)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_my_model():
    return load_model("model.keras")

model = load_my_model()

# ──────────────────────────────────────────────────────────────
# CANVAS
# stroke_width=25 → matches MNIST stroke density on a 280px canvas.
# MNIST strokes are ~12% of image width → 280 × 0.12 ≈ 34px.
# 25 is a comfortable drawing value that stays within that range.
# ──────────────────────────────────────────────────────────────
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=25,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# ──────────────────────────────────────────────────────────────
# PREPROCESSING
#
# Training pipeline (from your Colab):
#   X_train_sc = X_train / 255.0   ← only normalization, nothing else
#   No inversion. No StandardScaler. No reshape (CSV is already flat).
#   Pixel format: white digit = HIGH value, black background = LOW value.
#
# Canvas gives exactly this format:
#   White stroke on black bg → grayscale: digit=255, bg=0
#   After /255.0            → digit=1.0,  bg=0.0   ✓ matches training
#
# Extra steps (crop → center → blur) replicate the tight-crop
# centered format of real MNIST images so the model sees what
# it was trained on, not a raw 280×280→28×28 squash.
# ──────────────────────────────────────────────────────────────
def preprocess(image_data):
    # 1. RGBA → Grayscale
    #    White stroke [255,255,255,255] → 255
    #    Black bg     [0,0,0,255]       → 0
    img = cv2.cvtColor(image_data.astype("uint8"), cv2.COLOR_RGBA2GRAY)

    # 2. Binary threshold — remove canvas antialiasing noise
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 3. Find bounding box of the drawn digit (white pixels)
    #    NO INVERSION — white digit on black already matches training
    coords = np.column_stack(np.where(img > 0))
    if coords.shape[0] == 0:
        return None, None  # nothing drawn

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 4. Tight crop around digit only
    img = img[y_min:y_max + 1, x_min:x_max + 1]
    h, w = img.shape

    # 5. Resize to 20×20 preserving aspect ratio (MNIST convention)
    if w >= h:
        new_w = 20
        new_h = max(1, int(h * 20 / w))
    else:
        new_h = 20
        new_w = max(1, int(w * 20 / h))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 6. Center-pad to 28×28 (exactly like MNIST)
    top    = (28 - new_h) // 2
    bottom = 28 - new_h - top
    left   = (28 - new_w) // 2
    right  = 28 - new_w - left
    img = np.pad(img, ((top, bottom), (left, right)),
                 mode="constant", constant_values=0)

    # 7. Gaussian blur — softens hard binary edges to match
    #    MNIST's naturally anti-aliased handwritten strokes
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    # 8. Normalize — identical to training: pixel / 255.0
    img = img.astype("float32") / 255.0

    # Return display image (28×28) and flattened model input (1×784)
    return img, img.reshape(1, 784)


# ──────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    predict_clicked = st.button("Predict", use_container_width=True)
with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()

if predict_clicked:
    if canvas_result.image_data is None:
        st.warning("Please draw a digit first!")
    else:
        display_img, model_input = preprocess(canvas_result.image_data)

        if model_input is None:
            st.warning("Canvas is empty — please draw a digit!")
        else:
            # Show the exact 28×28 image the model receives
            st.image(
                display_img,
                width=150,
                caption="Processed image sent to model (28×28)"
            )

            # Run prediction
            pred       = model.predict(model_input, verbose=0)
            digit      = int(np.argmax(pred))
            confidence = float(np.max(pred)) * 100

            # Result with confidence-based feedback
            if confidence >= 70:
                st.success(f"### Predicted: {digit}  —  {confidence:.1f}% confident")
            elif confidence >= 40:
                st.warning(f"### Predicted: {digit}  —  {confidence:.1f}% (try drawing more clearly)")
            else:
                st.error(f"### Predicted: {digit}  —  {confidence:.1f}% (low confidence, please redraw)")

            # Per-class probability bar chart
            st.write("**Probability for each digit:**")
            chart_data = {str(i): float(pred[0][i]) for i in range(10)}
            st.bar_chart(chart_data)

# ──────────────────────────────────────────────────────────────
# TIPS
# ──────────────────────────────────────────────────────────────
with st.expander("Tips for best accuracy"):
    st.markdown("""
- **Draw large** — fill most of the black canvas
- **Center your digit** — MNIST digits are centered in the image  
- **Draw at normal speed** — rushed thin strokes reduce accuracy
- **One digit only** — the model expects a single digit per prediction
- If confidence is low, click **Clear** and try again
""")