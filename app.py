import gradio as gr
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO

# ZeroGPU support on Hugging Face Spaces (with graceful fallback for local CPU/GPU)
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# Select computing device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model when the application starts
model = YOLO("best.pt")
if device == "cuda":
    model.to("cuda")

def predict_fn(image, confidence):
    if image is None:
        return None, "Please upload a road image."

    image_array = np.array(image.convert("RGB"))

    results = model.predict(
        source=image_array,
        conf=float(confidence),
        iou=0.4,
        verbose=False,
        device=device
    )

    result = results[0]

    # Render bounding boxes on image
    annotated = result.plot()
    annotated = Image.fromarray(annotated[:, :, ::-1])

    if len(result.boxes) == 0:
        report = "No road damage detected."
    else:
        report = f"Total detections: {len(result.boxes)}\n\n"

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            score = float(box.conf[0])
            class_name = result.names[class_id]

            report += (
                f"{i+1}. {class_name}\n"
                f"Confidence: {score:.1%}\n\n"
            )

    return annotated, report

# Apply @spaces.GPU decorator if running on Hugging Face ZeroGPU
if has_spaces:
    detect_road_damage = spaces.GPU(predict_fn)
else:
    detect_road_damage = predict_fn

app = gr.Interface(
    fn=detect_road_damage,
    inputs=[
        gr.Image(type="pil", label="Upload Road Image"),
        gr.Slider(
            minimum=0.10,
            maximum=0.90,
            value=0.10,
            step=0.05,
            label="Confidence Threshold"
        )
    ],
    outputs=[
        gr.Image(type="pil", label="Detection Result"),
        gr.Textbox(label="Detection Report", lines=12)
    ],
    title="ROAD DAMAGE DETECTION SYSTEM",
    description="Upload a road image to detect cracks and potholes using YOLOv8.",
    submit_btn="Detect Road Damage"
)

if __name__ == "__main__":
    app.launch()
