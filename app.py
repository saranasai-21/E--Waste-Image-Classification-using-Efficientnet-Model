import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
import numpy as np
from PIL import Image

# =========================
# MIXED PRECISION POLICY
# =========================
mixed_precision.set_global_policy("mixed_float16")

# =========================
# CLASS NAMES
# =========================
CLASS_NAMES = [
    "Battery",
    "Keyboard",
    "Microwave",
    "Mobile",
    "Mouse",
    "PCB",
    "Player",
    "Printer",
    "Television",
    "Washing Machine"
]

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "ewaste_efficientnetv2b2.keras"

model = keras.models.load_model(
    MODEL_PATH,
    compile=False
)

# =========================
# IMAGE SIZE
# =========================
IMG_SIZE = 260

# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(image):

    if image is None:
        return None, "Please upload an image."

    try:
        # Convert to RGB
        image = image.convert("RGB")

        # Resize
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # Convert to numpy
        img_array = np.array(image)

        # Expand dims
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        # Convert float16 -> float32
        predictions = predictions.astype(np.float32)

        # Get class index
        predicted_index = np.argmax(predictions)

        # Get confidence
        confidence = float(np.max(predictions)) * 100

        # Get label
        predicted_label = CLASS_NAMES[predicted_index]

        # Probabilities dictionary
        probs = {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }

        result_text = f"""
✅ Prediction: {predicted_label}

🎯 Confidence: {confidence:.2f}%
"""

        return probs, result_text

    except Exception as e:
        return None, f"Error: {str(e)}"

# =========================
# CUSTOM CSS
# =========================
custom_css = """
body {
    background: #0f172a;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

.support-box {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #374151;
}

.support-title {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    color: white;
    margin-bottom: 20px;
}

.class-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 15px;
}

.class-item {
    background: #1e293b;
    padding: 12px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: bold;
    border: 1px solid #334155;
}

.upload-box {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #374151;
}

.result-box {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid #374151;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 20px;
}
"""

# =========================
# UI
# =========================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="main-title">
        ♻️ E-Waste Image Classifier
    </div>

    <div class="subtitle">
        Upload an E-Waste image and classify it using EfficientNetV2B2
    </div>
    """)

    # =========================
    # SUPPORTED CLASSES
    # =========================
    gr.HTML("""
    <div class="support-box">
        <div class="support-title">
            Supported Classes
        </div>

        <div class="class-grid">
            <div class="class-item">🔋 Battery</div>
            <div class="class-item">⌨️ Keyboard</div>
            <div class="class-item">🍲 Microwave</div>
            <div class="class-item">📱 Mobile</div>
            <div class="class-item">🖱️ Mouse</div>

            <div class="class-item">💻 PCB</div>
            <div class="class-item">🎵 Player</div>
            <div class="class-item">🖨️ Printer</div>
            <div class="class-item">📺 Television</div>
            <div class="class-item">🧺 Washing Machine</div>
        </div>
    </div>
    """)

    gr.Markdown("")

    # =========================
    # UPLOAD + RESULT SIDE BY SIDE
    # =========================
    with gr.Row():

        with gr.Column():

            gr.HTML("""
            <div class="upload-box">
                <h2 style="color:white;text-align:center;">
                    Upload Image
                </h2>
            </div>
            """)

            image_input = gr.Image(
                type="pil",
                label="Upload E-Waste Image",
                height=400
            )

        with gr.Column():

            gr.HTML("""
            <div class="result-box">
                <h2 style="color:white;text-align:center;">
                    Prediction Results
                </h2>
            </div>
            """)

            label_output = gr.Label(
                num_top_classes=10,
                label="Class Probabilities"
            )

            text_output = gr.Textbox(
                label="Prediction",
                lines=5
            )

    # =========================
    # BUTTONS
    # =========================
    with gr.Row():

        predict_btn = gr.Button(
            "🚀 Predict",
            variant="primary"
        )

        clear_btn = gr.Button(
            "🗑️ Clear"
        )

    # =========================
    # BUTTON ACTIONS
    # =========================
    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[label_output, text_output]
    )

    clear_btn.click(
        fn=lambda: (None, None, ""),
        outputs=[image_input, label_output, text_output]
    )

    # =========================
    # FOOTER
    # =========================
    gr.HTML("""
    <div class="footer">
        Built with TensorFlow, Keras & Gradio
    </div>
    """)

# =========================
# LAUNCH
# =========================
demo.queue().launch(
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860))
)
