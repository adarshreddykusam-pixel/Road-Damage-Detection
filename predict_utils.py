import cv2
import numpy as np
from PIL import Image
import json
import os
import tempfile

# 5 Target Road Damage Classes with civil engineering hazard weights and standardized colors
CLASS_INFO = {
    0: {
        "name": "Longitudinal Crack",
        "weight": 1.0,
        "color_bgr": (255, 200, 0),       # Cyan / Sky Blue
        "hex": "#00C8FF",
        "description": "Crack parallel to centerline caused by joint deterioration or pavement fatigue."
    },
    1: {
        "name": "Transverse Crack",
        "weight": 1.2,
        "color_bgr": (0, 215, 255),       # Bright Yellow
        "hex": "#FFD700",
        "description": "Crack perpendicular to centerline caused by thermal shrinkage or base reflection."
    },
    2: {
        "name": "Alligator Crack",
        "weight": 2.2,
        "color_bgr": (0, 140, 255),       # Vibrant Orange
        "hex": "#FF8C00",
        "description": "Interconnected crack network indicating severe structural fatigue of the base layer."
    },
    3: {
        "name": "Other Corruption",
        "weight": 0.8,
        "color_bgr": (214, 112, 218),     # Lavender / Magenta
        "hex": "#DA70D6",
        "description": "Surface degradation, spalling, patchwork failure, or uneven road wear."
    },
    4: {
        "name": "Pothole",
        "weight": 3.5,
        "color_bgr": (46, 46, 255),        # Crimson Red (Highest Hazard)
        "hex": "#FF2E2E",
        "description": "Deep bowl-shaped road cavity posing immediate safety hazard to vehicles."
    }
}


def apply_clahe_enhancement(image_bgr):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the L-channel
    in LAB color space. Enhances cracks in shadows and glare on asphalt.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def annotate_frame(frame_bgr, boxes, names):
    """
    Renders high-visibility bounding boxes and styled labels on a frame.
    """
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = names.get(cls_id, f"Class {cls_id}")

        info = CLASS_INFO.get(cls_id, {
            "color_bgr": (0, 255, 0),
            "weight": 1.0
        })
        color = info["color_bgr"]

        # Coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        # Draw main bounding box with shadow for contrast
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 0), 3)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label badge
        label = f"{class_name} {conf:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        badge_y1 = max(0, y1 - label_h - 8)
        badge_y2 = y1
        badge_x2 = min(w, x1 + label_w + 10)

        # Semi-transparent tag background
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Dark text for light colors, light text for dark colors
        text_color = (0, 0, 0) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255)
        cv2.putText(annotated, label, (x1 + 4, badge_y2 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return annotated


def calculate_severity_index(boxes, frame_shape):
    """
    Calculates Road Distress Severity Score (0 - 100) based on detected damage classes,
    bounding box area proportion, and class hazard weights.
    """
    if len(boxes) == 0:
        return 0.0, "Good Condition", "Normal pavement. No visible distress detected. Routine monitoring recommended."

    h, w = frame_shape[:2]
    total_area = float(h * w)

    raw_score = 0.0
    has_pothole = False
    pothole_count = 0

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        info = CLASS_INFO.get(cls_id, {"weight": 1.0})
        weight = info["weight"]

        x1, y1, x2, y2 = map(float, box.xyxy[0])
        box_area = max(0.0, (x2 - x1) * (y2 - y1))
        area_ratio = box_area / total_area

        # Severity contribution: base weight + area scaling + confidence
        score_contrib = (weight * 15.0) + (area_ratio * 250.0 * weight) * conf
        raw_score += score_contrib

        if cls_id == 4:
            has_pothole = True
            pothole_count += 1

    # Additional penalty for multiple potholes
    if pothole_count > 1:
        raw_score += (pothole_count - 1) * 20.0

    severity_score = min(100.0, round(raw_score, 1))

    # Condition & Recommendation determination
    if severity_score >= 60.0 or pothole_count >= 2:
        condition = "Critical Failure (Severe Hazard)"
        recommendation = "High risk to vehicles! Immediate emergency pothole cold-patching and structural inspection required within 24-48 hours."
    elif severity_score >= 35.0 or has_pothole:
        condition = "Poor (High Distress)"
        recommendation = "Significant degradation detected. Prioritize localized patching, deep crack sealing, and structural resurfacing in next maintenance cycle."
    elif severity_score >= 15.0:
        condition = "Fair (Moderate Distress)"
        recommendation = "Moderate surface distress. Schedule preventive maintenance, crack sealing, and drainage audit to prevent pothole formation."
    else:
        condition = "Minor Wear (Low Distress)"
        recommendation = "Minor superficial cracking. Continue regular municipal monitoring."

    return severity_score, condition, recommendation


def predict_image(image, model, conf_thresh=0.20, iou_thresh=0.45, enhance_contrast=False, device="cpu"):
    """
    Runs full conference-level inference on a single image.
    Returns: annotated PIL image, markdown report, structured JSON log.
    """
    if image is None:
        return None, "Please upload a road image.", "{}"

    # Convert PIL to BGR OpenCV format
    img_array = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    if enhance_contrast:
        img_bgr = apply_clahe_enhancement(img_bgr)

    results = model.predict(
        source=img_bgr,
        conf=float(conf_thresh),
        iou=float(iou_thresh),
        verbose=False,
        device=device
    )

    result = results[0]
    boxes = result.boxes
    annotated_bgr = annotate_frame(img_bgr, boxes, result.names)

    # Convert back to PIL RGB for display
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Calculate severity score & maintenance recommendation
    severity_score, condition, recommendation = calculate_severity_index(boxes, img_bgr.shape)

    # Tally damage counts
    counts = {info["name"]: 0 for info in CLASS_INFO.values()}
    detections_log = []

    for i, box in enumerate(boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = result.names.get(cls_id, f"Class {cls_id}")
        counts[class_name] = counts.get(class_name, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections_log.append({
            "id": i + 1,
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": round(conf, 4),
            "bbox_xyxy": [x1, y1, x2, y2]
        })

    # Generate Markdown Inspection Report
    report = f"### 🛣️ Road Inspection & Distress Analysis\n\n"
    report += f"- **Distress Condition:** `{condition}`\n"
    report += f"- **Road Hazard Severity Index:** **{severity_score:.1f} / 100**\n"
    report += f"- **Total Defects Detected:** **{len(boxes)}**\n"
    report += f"- **Recommended Action:** {recommendation}\n\n"

    report += "#### 📋 Defect Breakdown\n"
    report += "| Damage Type | Count | Severity Weight | Status |\n"
    report += "|:---|:---:|:---:|:---|\n"
    for cls_id, info in CLASS_INFO.items():
        cnt = counts.get(info["name"], 0)
        status_badge = "🔴 Alert" if cnt > 0 and cls_id == 4 else ("🟠 Found" if cnt > 0 else "🟢 Clear")
        report += f"| {info['name']} | {cnt} | {info['weight']}x | {status_badge} |\n"

    report += "\n"

    json_payload = json.dumps({
        "total_detections": len(boxes),
        "severity_score": severity_score,
        "condition": condition,
        "recommendation": recommendation,
        "breakdown": counts,
        "detections": detections_log
    }, indent=2)

    return annotated_pil, report, json_payload


def predict_video(video_path, model, conf_thresh=0.20, iou_thresh=0.45, frame_stride=2, device="cpu", progress=None):
    """
    Processes video files (dashcam, survey footage) frame-by-frame,
    drawing bounding boxes, on-screen HUD, and generating temporal defect analytics.
    """
    if not video_path or not os.path.exists(video_path):
        return None, "Please upload a valid video file."

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video file."

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Effective FPS for output video considering stride
    effective_fps = max(1.0, orig_fps / max(1, frame_stride))

    # Temp file for processed video
    temp_dir = tempfile.gettempdir()
    out_video_path = os.path.join(temp_dir, f"road_damage_output_{os.path.basename(video_path)}")
    if not out_video_path.endswith(".mp4"):
        out_video_path += ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, effective_fps, (width, height))

    frame_idx = 0
    sampled_frames_count = 0
    damage_events = []
    cumulative_counts = {info["name"]: 0 for info in CLASS_INFO.values()}
    peak_severity = 0.0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            sampled_frames_count += 1
            timestamp_sec = frame_idx / orig_fps

            # YOLO inference
            results = model.predict(
                source=frame_bgr,
                conf=float(conf_thresh),
                iou=float(iou_thresh),
                verbose=False,
                device=device
            )
            result = results[0]
            boxes = result.boxes

            # Annotate bounding boxes
            annotated = annotate_frame(frame_bgr, boxes, result.names)

            # Compute frame severity
            frame_severity, _, _ = calculate_severity_index(boxes, frame_bgr.shape)
            peak_severity = max(peak_severity, frame_severity)

            # On-screen HUD Banner
            hud_text_left = f"Frame: {frame_idx}/{total_frames} | Time: {timestamp_sec:.1f}s"
            hud_text_right = f"Defects: {len(boxes)} | Hazard Index: {frame_severity:.0f}/100"

            cv2.rectangle(annotated, (0, 0), (width, 42), (20, 20, 20), -1)
            cv2.putText(annotated, hud_text_left, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, hud_text_right, (max(15, width - 420), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 215, 255), 2, cv2.LINE_AA)

            # Track events
            if len(boxes) > 0:
                frame_summary = []
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = result.names.get(cls_id, f"Class {cls_id}")
                    cumulative_counts[class_name] = cumulative_counts.get(class_name, 0) + 1
                    conf = float(box.conf[0])
                    frame_summary.append(f"{class_name} ({conf:.0%})")

                mins = int(timestamp_sec // 60)
                secs = timestamp_sec % 60
                damage_events.append(f"`[{mins:02d}:{secs:04.1f}]` " + ", ".join(frame_summary))

            writer.write(annotated)

        frame_idx += 1

    cap.release()
    writer.release()

    # Determine overall road condition from video survey
    total_detections = sum(cumulative_counts.values())
    if peak_severity >= 60.0 or cumulative_counts.get("Pothole", 0) >= 3:
        overall_condition = "Critical Failure — Immediate Action Required"
        overall_rec = "Multiple high-hazard potholes/distress detected along this route. Issue urgent repair work-order."
    elif peak_severity >= 30.0 or cumulative_counts.get("Pothole", 0) > 0:
        overall_condition = "Poor Road Quality — Resurfacing Needed"
        overall_rec = "Route shows progressive fatigue cracking and localized pothole hazards. Schedule asphalt resurfacing."
    elif total_detections > 5:
        overall_condition = "Fair — Preventive Maintenance Required"
        overall_rec = "Moderate cracking present. Apply sealant to prevent water infiltration and pothole genesis."
    else:
        overall_condition = "Good Condition — Routine Monitoring"
        overall_rec = "Road section is in satisfactory condition with minimal defects."

    # Generate Video Report
    duration_sec = total_frames / orig_fps if orig_fps > 0 else 0
    report = f"### 🎥 Video Road Survey & Temporal Analytics\n\n"
    report += f"- **Video Duration:** `{duration_sec:.1f} seconds` ({total_frames} total frames, {sampled_frames_count} analyzed)\n"
    report += f"- **Surveyed Condition Rating:** `{overall_condition}`\n"
    report += f"- **Peak Distress Index:** **{peak_severity:.1f} / 100**\n"
    report += f"- **Total Defects Logged:** **{total_detections}**\n"
    report += f"- **Municipal Action Recommendation:** {overall_rec}\n\n"

    report += "#### 📊 Route Defect Frequency\n"
    report += "| Defect Type | Total Detections | Hazard Level |\n"
    report += "|:---|:---:|:---|\n"
    for cls_id, info in CLASS_INFO.items():
        cnt = cumulative_counts.get(info["name"], 0)
        badge = "🔴 Critical" if cls_id == 4 and cnt > 0 else ("🟠 Warning" if cnt > 0 else "🟢 None")
        report += f"| {info['name']} | {cnt} | {badge} |\n"

    report += "\n#### ⏱️ Temporal Defect Timeline (Sampled Events)\n"
    if damage_events:
        for event in damage_events[:15]:  # Display up to 15 key events
            report += f"- {event}\n"
        if len(damage_events) > 15:
            report += f"- *...and {len(damage_events) - 15} more occurrences throughout the route.*\n"
    else:
        report += "- *No significant distress events observed along this video segment.*\n"

    return out_video_path, report
