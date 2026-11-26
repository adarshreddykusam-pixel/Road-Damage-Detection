"""
Road Damage Detection — Research Data Pipeline & Augmentation Suite
Designed for Academic Conference Benchmarking & Dataset Preprocessing.

Supports:
1. Contrast Limited Adaptive Histogram Equalization (CLAHE) for shadow & glare normalization.
2. Simulated Adverse Weather Augmentation (wet asphalt, rain reflections, motion blur).
3. Dataset Class Imbalance Analysis & Conference Paper Statistics Table Generation.
4. YOLO Format Dataset Verification & Bounding Box Quality Checks.
"""

import os
import cv2
import numpy as np
import yaml
from pathlib import Path


CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Other Corruption",
    "Pothole"
]


def enhance_road_clahe(image, clip_limit=2.5, tile_grid_size=(8, 8)):
    """
    Applies CLAHE on luminance channel to bring out road surface fissures
    obscured by tree shadows, underpasses, or direct sun glare.
    """
    if len(image.shape) == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)


def simulate_wet_asphalt(image):
    """
    Simulates wet pavement conditions by increasing specular highlights,
    lowering base diffuse reflectivity, and introducing localized specular glare.
    """
    img_float = image.astype(np.float32) / 255.0
    # Darken base asphalt slightly (absorption from wet surface)
    darkened = np.power(img_float, 1.15)
    # Add localized specular reflection gradient
    h, w = image.shape[:2]
    gradient = np.linspace(0.0, 0.25, h).reshape(h, 1, 1)
    wet_sim = np.clip((darkened + gradient * darkened) * 255.0, 0, 255).astype(np.uint8)
    return wet_sim


def simulate_motion_blur(image, kernel_size=15):
    """
    Simulates dashcam / vehicle motion blur caused by vehicle cruising speed.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)


def analyze_dataset_distribution(labels_dir, num_classes=5):
    """
    Parses YOLO annotation text files and generates comprehensive conference-ready
    statistics on class distributions, bounding box scales, and imbalance metrics.
    """
    labels_path = Path(labels_dir)
    if not labels_path.exists():
        return {"error": f"Path {labels_dir} does not exist."}

    class_counts = {i: 0 for i in range(num_classes)}
    bbox_areas = []
    aspect_ratios = []
    total_images_with_labels = 0
    total_boxes = 0

    for label_file in labels_path.glob("**/*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        if lines:
            total_images_with_labels += 1

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])

                if 0 <= cls_id < num_classes:
                    class_counts[cls_id] += 1
                    total_boxes += 1
                    bbox_areas.append(w * h)
                    aspect_ratios.append(w / max(1e-5, h))

    # Calculate class imbalance ratios (relative to majority class)
    max_count = max(class_counts.values()) if total_boxes > 0 else 1
    imbalance_ratios = {
        cls_id: round(max_count / max(1, count), 2)
        for cls_id, count in class_counts.items()
    }

    stats = {
        "total_annotated_files": total_images_with_labels,
        "total_bounding_boxes": total_boxes,
        "class_counts": {CLASS_NAMES[i]: class_counts[i] for i in range(num_classes)},
        "imbalance_ratios": {CLASS_NAMES[i]: imbalance_ratios[i] for i in range(num_classes)},
        "avg_bbox_area": round(float(np.mean(bbox_areas)), 4) if bbox_areas else 0.0,
        "avg_aspect_ratio": round(float(np.mean(aspect_ratios)), 2) if aspect_ratios else 0.0
    }
    return stats


def generate_conference_latex_table(stats):
    """
    Formats the dataset analysis into a clean LaTeX table suitable
    for inclusion in academic conference paper manuscripts (IEEE/CVPR/TRB format).
    """
    if "error" in stats:
        return stats["error"]

    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Road Damage Dataset Class Distribution and Imbalance Statistics}",
        "\\label{tab:road_damage_dist}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "\\textbf{Class ID} & \\textbf{Damage Category} & \\textbf{Instance Count} & \\textbf{Proportion (\\%)} & \\textbf{Imbalance Ratio} \\\\",
        "\\hline"
    ]

    total = max(1, stats["total_bounding_boxes"])
    for idx, name in enumerate(CLASS_NAMES):
        count = stats["class_counts"].get(name, 0)
        prop = (count / total) * 100.0
        ratio = stats["imbalance_ratios"].get(name, 1.0)
        latex.append(f"{idx} & {name} & {count} & {prop:.1f}\\% & {ratio:.1f}:1 \\\\")

    latex.extend([
        "\\hline",
        f"\\textbf{{Total}} & \\textbf{{All Categories}} & \\textbf{{{total}}} & \\textbf{{100.0\\%}} & - \\\\",
        "\\hline",
        "\\end{tabular}",
        "\\end{table}"
    ])
    return "\n".join(latex)
