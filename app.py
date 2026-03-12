from ultralytics import YOLO
import gradio as gr
import numpy as np
from PIL import Image

model = YOLO("yolov8n.pt")  # auto-download weights

def detect_vehicles(img):
    img_array = np.array(img.convert("RGB"))
    results = model(img_array)
    annotated = results[0].plot()
    return Image.fromarray(annotated)

gr.Interface(
    fn=detect_vehicles,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Vehicle Detection Web App",
    description="Upload an image to detect vehicles (YOLOv8)."
).launch()
