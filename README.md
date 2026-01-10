# Automated Road Damage Detection using YOLOv8
### Crackathon (IIT Bombay) – Team Project

This repository contains an end-to-end **computer vision pipeline for automated road damage detection**, developed as part of **Crackathon (IIT Bombay)**.  
The system is designed to handle **large-scale, real-world road imagery** and produce **deployment-ready detection outputs**.

---

## Problem Statement
Manual road inspection is slow, subjective, and non-scalable for modern infrastructure systems.  
This project aims to **automate the detection and localization of road surface damage** using deep learning–based object detection.

---

## Dataset
**Road Damage Detection 2022 (RDD2022)**

- ~47,000+ high-resolution annotated road images  
- Multi-national data collected under real-world conditions  
- Significant class imbalance and high visual variance  

### Damage Classes
| Class ID | Damage Type |
|--------|------------|
| 0 | Longitudinal Crack |
| 1 | Transverse Crack |
| 2 | Alligator Crack |
| 3 | Other Corruption |
| 4 | Pothole |

### Annotation Format
YOLO format with normalized coordinates:

<class_id> <x_center> <y_center> <width> <height>
---

## Model Architecture
**YOLOv8m (Ultralytics)**

Chosen for:
- Balance between accuracy and inference speed  
- Single-stage detection suitable for real-time systems  
- Decoupled detection head for stable optimization  
- COCO-pretrained weights for faster convergence  

---

## Training Strategy

### Configuration
| Parameter | Value |
|--------|------|
| Model | YOLOv8m |
| Input Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Epochs | 20–30 |
| Losses | Box, Classification, DFL |

Transfer learning was used to stabilize training on a large, heterogeneous dataset.

---

## Data Augmentation
To improve generalization across diverse road conditions:
- Mosaic augmentation  
- HSV color jittering  
- CLAHE (contrast normalization)  
- Random horizontal flipping  
- Gaussian blur  

---

## Training Challenges & Mitigation
**Challenges**
- Severe class imbalance  
- High intra-class variance (lighting, textures)  
- GPU memory constraints  
- Ephemeral cloud environments  

**Mitigation**
- Careful batch sizing  
- Early stopping with validation monitoring  
- Immediate persistence of trained artifacts  
- Reproducible training configuration  

---

## Evaluation
Performance evaluated using **Mean Average Precision (mAP)**:
- **mAP@0.5**
- **mAP@0.5:0.95**

### Results
- mAP@0.5 ≈ **0.55 – 0.60** on validation data  
- Stable localization across all damage categories  

---

## Inference & Outputs
- ~26K+ test images processed  
- One prediction file per image  

### Output Format
<class_id> <x_center> <y_center> <width> <height> <confidence>

### Output Structure
└── predictions/
    ├── 000001.txt
    ├── 000002.txt
    ├── 000003.txt
    └── ...

Outputs are structured for direct integration into analytics or inspection pipelines.


---

### Repository Structure
├── data.yaml
├── training_notebook.ipynb
├── predictions/
│   ├── *.txt
├── best.pt
└── README.md



---

## Team
This project was developed as a **two-member team** during Crackathon (IIT Bombay).

- **Contributor:** NISHANT TRIPATHI
- **Contributor:** OJASVI MARWARI

---

## Future Work
- Damage severity estimation  
- Video-based temporal analysis  
- Edge-device deployment  
- Integration with GIS / smart-city platforms  

---

## License
This project is intended for **educational and research purposes**.

---

## Contact
Open to discussions and opportunities in **Machine Learning / Computer Vision / AI Engineering**.
