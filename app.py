import os
import gradio as gr
from PIL import Image
import torch
from ultralytics import YOLO

import predict_utils

# ZeroGPU support on Hugging Face Spaces (with graceful fallback for local CPU/CUDA)
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False

# Select available hardware acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 fine-tuned road damage weights
model_path = "best.pt"
if not os.path.exists(model_path):
    # Fallback to yolov8n.pt if best.pt is absent
    model_path = "yolov8n.pt"

model = YOLO(model_path)
if device == "cuda":
    model.to("cuda")


def run_image_inference(image, conf_thresh, iou_thresh, enhance_contrast):
    """Handler for image inspection tab"""
    if image is None:
        return None, "⚠️ Please upload a road image to begin inspection.", "{}"
    
    annotated_img, report, json_log = predict_utils.predict_image(
        image=image,
        model=model,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        enhance_contrast=enhance_contrast,
        device=device
    )
    return annotated_img, report, json_log


def run_video_inference(video_path, conf_thresh, frame_stride):
    """Handler for video inspection tab"""
    if not video_path:
        return None, "⚠️ Please upload a dashcam or road survey video file."
    
    out_video, report = predict_utils.predict_video(
        video_path=video_path,
        model=model,
        conf_thresh=conf_thresh,
        iou_thresh=0.45,
        frame_stride=int(frame_stride),
        device=device
    )
    return out_video, report


# Wrap GPU execution functions for Hugging Face ZeroGPU
if has_spaces:
    infer_image = spaces.GPU(run_image_inference)
    infer_video = spaces.GPU(run_video_inference)
else:
    infer_image = run_image_inference
    infer_video = run_video_inference


# Build Gradio Multi-Tab Interface
with gr.Blocks(title="Road Damage Detection System (Image & Video)") as app:
    gr.Markdown(
        """
        # 🛣️ Automated Road Damage Detection & Distress Assessment System
        ### YOLOv8 Computer Vision Pipeline for Real-Time Infrastructure Auditing
        *Accepts **Single Road Imagery** and **Dashcam Video Files** with automated severity indexing and maintenance recommendations.*
        """
    )

    with gr.Tabs():
        # TAB 1: IMAGE INSPECTION
        with gr.TabItem("📷 Image Inspection"):
            with gr.Row():
                with gr.Column(scale=5):
                    image_input = gr.Image(type="pil", label="Upload Road Image (JPG, PNG, WebP)")
                    with gr.Accordion("⚙️ Inference & Preprocessing Settings", open=True):
                        conf_slider = gr.Slider(
                            minimum=0.05,
                            maximum=0.90,
                            value=0.15,
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        iou_slider = gr.Slider(
                            minimum=0.20,
                            maximum=0.70,
                            value=0.45,
                            step=0.05,
                            label="IoU NMS Threshold"
                        )
                        clahe_check = gr.Checkbox(
                            value=False,
                            label="Apply CLAHE Enhancement (Improves visibility under shadows & asphalt glare)"
                        )
                    detect_img_btn = gr.Button("🔍 Detect Road Damage", variant="primary", size="lg")

                with gr.Column(scale=6):
                    image_output = gr.Image(type="pil", label="Annotated Detection Output")
                    report_output = gr.Markdown(label="Inspection Report")
                    with gr.Accordion("📄 Structured JSON Inspection Log", open=False):
                        json_output = gr.Code(label="Exportable Payload", language="json")

            detect_img_btn.click(
                fn=infer_image,
                inputs=[image_input, conf_slider, iou_slider, clahe_check],
                outputs=[image_output, report_output, json_output]
            )

        # TAB 2: VIDEO SURVEY
        with gr.TabItem("🎥 Video & Dashcam Survey"):
            with gr.Row():
                with gr.Column(scale=5):
                    video_input = gr.Video(label="Upload Dashcam / Inspection Video (MP4, MOV, AVI)")
                    with gr.Accordion("⚙️ Video Processing Parameters", open=True):
                        vid_conf_slider = gr.Slider(
                            minimum=0.10,
                            maximum=0.80,
                            value=0.20,
                            step=0.05,
                            label="Confidence Threshold"
                        )
                        stride_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=2,
                            step=1,
                            label="Frame Stride (e.g. 2 = process every 2nd frame for 2x faster survey)"
                        )
                    detect_vid_btn = gr.Button("🎬 Process Route Video", variant="primary", size="lg")

                with gr.Column(scale=6):
                    video_output = gr.Video(label="Annotated Route Survey Video (with HUD)")
                    vid_report_output = gr.Markdown(label="Route Defect & Temporal Analytics")

            detect_vid_btn.click(
                fn=infer_video,
                inputs=[video_input, vid_conf_slider, stride_slider],
                outputs=[video_output, vid_report_output]
            )

        # TAB 3: CONFERENCE RESEARCH & METHODOLOGY
        with gr.TabItem("📊 Research & Benchmark Details"):
            gr.Markdown(
                """
                ### 🔬 Research Methodology & Taxonomy

                #### 1. Damage Class Taxonomy & Hazard Weighting
                Our model classifies distress based on structural severity and vehicular risk:
                - **Pothole (Class 4)**: High Hazard (`Weight: 3.5x`). Bowl-shaped cavities that pose immediate danger of tire puncture and vehicle loss-of-control.
                - **Alligator Crack (Class 2)**: Structural Hazard (`Weight: 2.2x`). Interconnected fatigue fissures indicating structural failure of base course.
                - **Transverse Crack (Class 1)**: Thermal Shrinkage (`Weight: 1.2x`). Perpendicular cracks that permit moisture infiltration into road base.
                - **Longitudinal Crack (Class 0)**: Longitudinal Joint Fatigue (`Weight: 1.0x`). Linear cracks along traffic direction.
                - **Other Corruption (Class 3)**: Superficial (`Weight: 0.8x`). Surface spalling, edge crumbling, and minor asphalt patchwork degradation.

                #### 2. Road Hazard Severity Index Formulation
                The system computes a normalized Road Distress Severity Index ($S \in [0, 100]$):
                $$\\text{Severity Index} = \\min\\left(100, \\sum_{i=1}^N \\left[ w_{\\text{class}} \\cdot 15 + \\left(\\frac{\\text{Area}_i}{\\text{Area}_{\\text{frame}}} \\cdot 250 \\cdot w_{\\text{class}}\\right) \\cdot \\text{conf}_i \\right] + P_{\\text{pothole}}\\right)$$
                Where $w_{\\text{class}}$ is the hazard weight, and $P_{\\text{pothole}}$ applies an exponential penalty for multiple concurrent potholes.

                #### 3. Continuous Video Auditing Architecture
                In video processing mode, the pipeline performs:
                1. Stride-based frame decimation to achieve real-time throughput.
                2. Real-time heads-up display (HUD) rendering timestamps and active hazard counts.
                3. Temporal event tracking logging the exact video timestamps of pavement distress.
                """
            )

if __name__ == "__main__":
    app.launch()
