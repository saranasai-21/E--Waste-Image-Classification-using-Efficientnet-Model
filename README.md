# ♻️ E-Waste Image Classification using EfficientNetV2B2

An image classification project that identifies different types of electronic waste using a fine-tuned EfficientNetV2B2 model. The application is deployed on Hugging Face Spaces and provides real-time predictions through an interactive Gradio interface.

## Live Webapp

🔗 Hugging Face Space:
https://huggingface.co/spaces/saranasai/e-waste-image-classification

---

## Project Overview

Electronic waste is one of the fastest-growing waste streams worldwide. Proper classification of e-waste is essential for efficient recycling and disposal.

This project uses Transfer Learning with EfficientNetV2B2 to classify images of electronic waste into 10 categories. Users can upload an image through a web interface and instantly receive a prediction along with confidence scores and class probabilities.

---

## Supported Classes

* Battery
* Keyboard
* Microwave
* Mobile
* Mouse
* PCB
* Player
* Printer
* Television
* Washing Machine

---

## Model Details

* Architecture: EfficientNetV2B2
* Framework: TensorFlow / Keras
* Learning Approach: Transfer Learning
* Number of Classes: 10
* Deployment: Hugging Face Spaces
* Interface: Gradio

---

## Results

The model was evaluated on a test set containing 300 images across 10 e-waste categories.

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 96.67% |
| Precision | 96.71% |
| Recall    | 96.67% |
| F1-Score  | 96.64% |

### Class-wise Performance

<img width="466" height="410" alt="Screenshot 2026-05-28 094451" src="https://github.com/user-attachments/assets/04144ea8-0ee5-43d4-a064-6f273d944d2e" />

The model achieved an overall accuracy of **96.67%**, demonstrating strong performance across all supported e-waste categories.


---

## Features

* Upload an e-waste image
* Real-time classification
* Confidence score display
* Class probability visualization
* Interactive Gradio interface
* Cloud deployment using Hugging Face Spaces

---

## Tech Stack

* Python
* TensorFlow
* Keras
* Gradio
* NumPy
* Pandas
* Matplotlib
* Hugging Face Spaces

---

## Application User Interface


<img width="936" height="836" alt="Screenshot 2026-05-29 132454" src="https://github.com/user-attachments/assets/f6ed5695-5a6b-4b6b-a520-7e55826e007f" />

---

## Results

<img width="970" height="618" alt="Screenshot 2026-05-29 132652" src="https://github.com/user-attachments/assets/8aa3a1a9-80fc-45ef-94c6-d8e345e306d2" />


---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/saranasai-21/E--Waste-Image-Classification-using-Efficientnet-Model.git
cd E--Waste-Image-Classification-using-Efficientnet-Model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## Future Improvements

* Add more e-waste categories
* Webcam-based classification
* Explainable AI visualizations (Grad-CAM)
* Mobile-friendly deployment
* Model optimization for faster inference
