import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gradio as gr
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
from keras.applications.efficientnet_v2 import preprocess_input
import numpy as np
from PIL import Image

# ==========================================
# CPU SETTINGS
# ==========================================
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ==========================================
# MIXED PRECISION SUPPORT
# ==========================================
try:
    mixed_precision.set_global_policy("mixed_float16")
except:
    pass

# ==========================================
# CLASS NAMES
# ==========================================
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

# ==========================================
# SETTINGS
# ==========================================
IMG_SIZE = 260

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_PATH = "ewaste_efficientnetv2b2.keras"

print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print("Model loaded successfully")
print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict_image(image):

    if image is None:
        return {}, "Please upload an image."

    try:

        # Convert image
        image = image.convert("RGB")

        # Resize
        image = image.resize((IMG_SIZE, IMG_SIZE))

        # To numpy
        img_array = np.array(image).astype(np.float32)

        # Batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # EfficientNetV2 preprocessing
        img_array = preprocess_input(img_array)

        # Predict
        predictions = model.predict(img_array, verbose=0)

        predictions = np.array(predictions).astype(np.float32)

        print("\n====================")
        print("Prediction Vector:")
        print(predictions)
        print("====================\n")

        # NaN protection
        if np.isnan(predictions).any():
            return {}, "Model returned invalid predictions (NaN)."

        predicted_index = int(np.argmax(predictions[0]))

        confidence = float(predictions[0][predicted_index]) * 100

        predicted_label = CLASS_NAMES[predicted_index]

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

        import traceback
        traceback.print_exc()

        return {}, f"Error: {str(e)}"

# ==========================================
# CUSTOM CSS
# ==========================================
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

# ==========================================
# UI
# ==========================================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div class="main-title">
        ♻️ E-Waste Image Classifier
    </div>
    <div class="subtitle">
        Upload an E-Waste image and classify it using EfficientNetV2B2
    </div>
    """)

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

    with gr.Row():

        predict_btn = gr.Button(
            "🚀 Predict",
            variant="primary"
        )

        clear_btn = gr.Button("🗑️ Clear")

    predict_btn.click(
        fn=predict_image,
        inputs=image_input,
        outputs=[label_output, text_output]
    )

    clear_btn.click(
        fn=lambda: (None, {}, ""),
        outputs=[image_input, label_output, text_output]
    )

    gr.HTML("""
    <div class="footer">
        Built with TensorFlow, Keras & Gradio
    </div>
    """)

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=7860
)
