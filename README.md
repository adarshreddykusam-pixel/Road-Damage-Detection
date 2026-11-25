# Road Damage Detection

Live Demo: [Hugging Face Space](https://huggingface.co/spaces/adarshreddykusam/road-damage-detection)

An automated deep learning system for road damage detection and classification using YOLOv8, integrated with an interactive Gradio web application.

## Overview
This system identifies and classifies road surface defects from images:
- Longitudinal Crack
- Transverse Crack
- Alligator Crack
- Other Corruption
- Pothole

## Features
- **Object Detection**: Fast bounding box predictions via fine-tuned YOLOv8.
- **Interactive Web App**: Upload road photos, tweak confidence thresholds, and review detection reports.
- **ZeroGPU Acceleration**: Runs with GPU acceleration on Hugging Face Spaces and supports CPU/GPU fallback locally.

## Setup & Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Locally
```bash
python app.py
```
Then open the local URL in your browser.

### 3. Python API
```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(source="image.jpg", conf=0.2)
results[0].show()
```

## Structure
- `app.py`: Gradio web interface.
- `best.pt`: Fine-tuned YOLOv8 model weights.
- `requirements.txt`: Project dependencies.
- `road_damage_data.yaml`: Dataset configuration.
- `road_damage_kaggle_training.ipynb`: Training notebook.
- `crackathon.ipynb`: Evaluation and data exploration notebook.
