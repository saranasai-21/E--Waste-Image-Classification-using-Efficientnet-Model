import gradio as gr
import tensorflow as tf
import numpy as np

# =====================================================
# Load Model
# =====================================================

model = tf.keras.models.load_model(
    "ewaste_efficientnetv2b2.keras",
    compile=False
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

    # No image uploaded
    if image is None:
        return {
            "Please Upload an Image": 1.0
        }

    # Convert image
    image = image.convert("RGB")

    # Resize
    image = image.resize((IMG_SIZE, IMG_SIZE))

    # Convert to array
    image_array = np.array(image)

    # Expand dimensions
    image_input = np.expand_dims(image_array, axis=0)

    # Preprocess
    image_input = tf.keras.applications.efficientnet_v2.preprocess_input(
        image_input
    )

    # Predict
    predictions = model.predict(image_input, verbose=0)[0]

    # Get highest confidence
    max_confidence = float(np.max(predictions))

    # Predicted class index
    predicted_index = np.argmax(predictions)

    # Predicted class name
    predicted_class = class_names[predicted_index]

    # =================================================
    # Validation Logic
    # =================================================

    # If confidence below 50%
    if max_confidence < 0.50:

        return {
            "❌ Not an E-Waste Image / Blurry / Multiple Items": 1.0
        }

    # =================================================
    # Valid Prediction
    # =================================================

    confidences = {
        predicted_class: max_confidence
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
    max-width: 1300px !important;
    margin: auto;
    padding-top: 20px;
    font-family: Arial, sans-serif;
}

/* ================================================== */
/* Header */
/* ================================================== */

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
    margin-bottom: 35px;
}

/* ================================================== */
/* Cards */
/* ================================================== */

.upload-card,
.result-card {
    background: #111827 !important;
    border-radius: 22px !important;
    padding: 20px !important;
    min-height: 560px !important;
    height: 100%;
    border: 1px solid #1e293b;
}

/* ================================================== */
/* Buttons */
/* ================================================== */

button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    height: 52px;
}

/* ================================================== */
/* Footer */
/* ================================================== */

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
    # Upload + Prediction Results
    # =================================================

    with gr.Row(equal_height=True):

        # =============================================
        # Upload Section
        # =============================================

        with gr.Column(scale=1):

            with gr.Group(elem_classes="upload-card"):

                gr.Markdown(
                    """
                    <div style="
                        color:white;
                        font-size:22px;
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
                    height=430,
                    container=False
                )

        # =============================================
        # Prediction Results
        # =============================================

        with gr.Column(scale=1):

            with gr.Group(elem_classes="result-card"):

                gr.Markdown(
                    """
                    <div style="
                        color:white;
                        font-size:22px;
                        font-weight:700;
                        margin-bottom:18px;
                    ">
                        📊 Prediction Results
                    </div>
                    """
                )

                output = gr.Label(
                    num_top_classes=1,
                    container=False
                )

                # Empty spacing
                gr.HTML(
                    """
                    <div style="height:330px;"></div>
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