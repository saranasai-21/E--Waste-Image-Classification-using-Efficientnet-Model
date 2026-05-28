import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# =====================================================
# Load Original Model
# =====================================================

model = load_model(
    "ewaste_efficientnetv2b2.keras",
    compile=False,
    safe_mode=False
)

# =====================================================
# Class Names
# =====================================================

class_names = [
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

IMG_SIZE = 260

# =====================================================
# Prediction Function
# =====================================================

def predict_image(image):

    if image is None:
        return {"Please Upload an Image": 1.0}

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to numpy
    image = np.array(image).astype("float32")

    # Expand dimensions
    image = np.expand_dims(image, axis=0)

    # Preprocess
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)

    # Predict
    predictions = model.predict(image, verbose=0)[0]

    # Convert predictions to dictionary
    confidences = {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

    return confidences


# =====================================================
# Custom CSS
# =====================================================

custom_css = """
body {
    background: #0f172a;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto;
    padding-top: 20px;
    font-family: Arial, sans-serif;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 30px;
}

.upload-card,
.result-card {
    background: #111827 !important;
    border-radius: 20px !important;
    padding: 20px !important;
    min-height: 520px !important;
    height: 100%;
}

button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    height: 52px;
}

.footer-text {
    text-align: center;
    margin-top: 25px;
    color: #94a3b8;
    font-size: 14px;
}
"""

# =====================================================
# UI
# =====================================================

with gr.Blocks(
    title="AI E-Waste Classifier",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate"
    ),
    css=custom_css
) as demo:

    # =================================================
    # Header
    # =================================================

    gr.Markdown(
        """
        <div class="main-title">
            ♻️ AI E-Waste Classifier
        </div>

        <div class="sub-title">
            Upload an electronic waste image and classify it using EfficientNetV2B2
        </div>
        """
    )

    # =================================================
    # Supported Classes
    # =================================================

    gr.HTML(
        """
        <div style="
            background:#111827;
            padding:22px;
            border-radius:20px;
            margin-bottom:30px;
            color:white;
        ">

        <h2 style="margin-bottom:18px;">
            📦 Supported Classes
        </h2>

        <div style="
            display:flex;
            justify-content:space-between;
            font-size:17px;
            font-weight:500;
            margin-bottom:15px;
            flex-wrap:wrap;
            gap:10px;
        ">
            <span>🔋 Battery</span>
            <span>•</span>
            <span>⌨️ Keyboard</span>
            <span>•</span>
            <span>📡 Microwave</span>
            <span>•</span>
            <span>📱 Mobile</span>
            <span>•</span>
            <span>🖱️ Mouse</span>
        </div>

        <div style="
            display:flex;
            justify-content:space-between;
            font-size:17px;
            font-weight:500;
            flex-wrap:wrap;
            gap:10px;
        ">
            <span>💻 PCB</span>
            <span>•</span>
            <span>🎵 Player</span>
            <span>•</span>
            <span>🖨️ Printer</span>
            <span>•</span>
            <span>📺 Television</span>
            <span>•</span>
            <span>🧺 Washing Machine</span>
        </div>

        </div>
        """
    )

    # =================================================
    # Upload + Prediction Results
    # =================================================

    with gr.Row(equal_height=True):

        # Upload Section

        with gr.Column(scale=1):

            with gr.Group(elem_classes="upload-card"):

                gr.Markdown(
                    """
                    <div style="
                        color:white;
                        font-size:20px;
                        font-weight:700;
                        margin-bottom:18px;
                    ">
                        📤 Upload E-Waste Image
                    </div>
                    """
                )

                image_input = gr.Image(
                    type="pil",
                    sources=["upload", "clipboard", "webcam"],
                    image_mode="RGB",
                    height=420,
                    container=False
                )

        # Prediction Results

        with gr.Column(scale=1):

            with gr.Group(elem_classes="result-card"):

                gr.Markdown(
                    """
                    <div style="
                        color:white;
                        font-size:20px;
                        font-weight:700;
                        margin-bottom:18px;
                    ">
                        📊 Prediction Results
                    </div>
                    """
                )

                output = gr.Label(
                    num_top_classes=5,
                    container=False
                )

                gr.HTML(
                    """
                    <div style="height:320px;"></div>
                    """
                )

    # =================================================
    # Buttons
    # =================================================

    with gr.Row():

        predict_btn = gr.Button(
            "🔍 Predict",
            variant="primary"
        )

        clear_btn = gr.ClearButton(
            [image_input, output],
            value="🗑 Clear"
        )

    # =================================================
    # Prediction Event
    # =================================================

    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=output
    )

    # =================================================
    # Footer
    # =================================================

    gr.Markdown(
        """
        <div class="footer-text">
            Built with TensorFlow • EfficientNetV2B2 • Gradio
        </div>
        """
    )

# =====================================================
# Launch
# =====================================================

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
