# 🛣️ Automated Road Damage Detection System (Image & Video)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/adarshreddykusam/road-damage-detection)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/adarshreddykusam-pixel/Road-Damage-Detection)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Gradio UI](https://img.shields.io/badge/Gradio-Multi--Modal%20UI-orange?style=for-the-badge&logo=gradio)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

An end-to-end computer vision pipeline for **automated road damage detection and pavement distress assessment**, powered by fine-tuned **YOLOv8** and deployed as an interactive web application on **Hugging Face Spaces**.

Designed for municipal road auditing, autonomous inspection vehicles, and academic research, the system accepts both **single road imagery** and **continuous dashcam/drone video streams**.

---

## 🚀 Live Demo

Try the live system directly in your browser:  
👉 **[Launch Hugging Face Space](https://huggingface.co/spaces/adarshreddykusam/road-damage-detection)**

---

## 🌟 Key Features

- **Dual-Modal Inputs**:
  - 📷 **Single Road Images**: Detailed bounding box localization, defect breakdown, and severity assessment.
  - 🎥 **Continuous Video / Dashcam Surveys**: Real-time HUD banner with frame timestamps, active hazard counts, and route distress event timelines.
- **5 Classified Damage Types**:
  - `Longitudinal Crack` (parallel to road direction)
  - `Transverse Crack` (perpendicular thermal fissures)
  - `Alligator Crack` (structural base fatigue network)
  - `Other Corruption` (surface spalling and degradation)
  - `Pothole` (high-hazard cavities posing vehicular risk)
- **Road Hazard Severity Index**: Automatically computes a normalized distress score ($0 - 100$) weighted by defect area and hazard severity to recommend targeted maintenance interventions (e.g. *Cold Patching*, *Crack Sealing*, *Resurfacing*).
- **Adverse Lighting Enhancement (CLAHE)**: Optional Contrast Limited Adaptive Histogram Equalization to penetrate harsh shadows and asphalt glare.
- **Municipal GIS Export**: Generates structured JSON payloads containing normalized bounding boxes, confidences, and damage classes.

---

## 🎯 Damage Classification & Hazard Weights

| Class ID | Damage Category | Hazard Weight | Civil Engineering Impact | Standard Color |
|:---:|:---|:---:|:---|:---:|
| `0` | **Longitudinal Crack** | `1.0x` | Cracks parallel to centerline from joint fatigue or pavement aging. | Sky Blue (`#00C8FF`) |
| `1` | **Transverse Crack** | `1.2x` | Thermal shrinkage cracks perpendicular to road centerline. | Yellow (`#FFD700`) |
| `2` | **Alligator Crack** | `2.2x` | Interconnected fissures indicating structural failure of base layers. | Orange (`#FF8C00`) |
| `3` | **Other Corruption** | `0.8x` | Surface spalling, crumbling, or minor asphalt patchwork wear. | Magenta (`#DA70D6`) |
| `4` | **Pothole** | `3.5x` | Dangerous road depression requiring urgent municipal repair. | Red (`#FF2E2E`) |

---

## 🛠️ Quick Start & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/adarshreddykusam-pixel/Road-Damage-Detection.git
cd Road-Damage-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python app.py
```
Open `http://127.0.0.1:7860` to access the interactive web interface.

---

## 🐍 Programmatic Usage (Python)

### Image Inference
```python
from PIL import Image
from ultralytics import YOLO
import predict_utils

model = YOLO("best.pt")
img = Image.open("road_photo.jpg")

annotated_img, report, json_log = predict_utils.predict_image(
    image=img,
    model=model,
    conf_thresh=0.15,
    enhance_contrast=True
)

annotated_img.save("annotated_output.jpg")
print(report)
```

### Video Inference
```python
from ultralytics import YOLO
import predict_utils

model = YOLO("best.pt")

output_video_path, video_report = predict_utils.predict_video(
    video_path="dashcam_footage.mp4",
    model=model,
    conf_thresh=0.20,
    frame_stride=2  # Process every 2nd frame for 2x speed
)

print(f"Processed video saved to: {output_video_path}")
print(video_report)
```

---

## 📂 Repository Structure

```plaintext
Road-Damage-Detection/
├── app.py                            # Gradio multi-modal web application (Image & Video)
├── predict_utils.py                  # Core inference, severity calculation & video HUD engine
├── data_pipeline.py                  # Conference dataset analysis & augmentation utilities
├── best.pt                           # Fine-tuned YOLOv8 model weights
├── requirements.txt                  # Python dependencies
├── road_damage_data.yaml             # YOLO dataset configuration
├── road_damage_kaggle_training.ipynb # End-to-end Kaggle training & benchmark notebook
├── crackathon.ipynb                  # Data exploration and evaluation notebook
├── road_damage_eda.png               # Exploratory data analysis charts
├── Project Report.pdf                # Technical project report
├── runs/detect/predict/              # Sample detection outputs
└── README.md                         # Documentation
```

---

## 👤 Author & Credits

**Adarsh Reddy Kusam**  
- Hugging Face: [@adarshreddykusam](https://huggingface.co/spaces/adarshreddykusam/road-damage-detection)  
- GitHub: [@adarshreddykusam-pixel](https://github.com/adarshreddykusam-pixel)  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
