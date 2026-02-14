#!/usr/bin/env python3
"""
Depth Pro - Local Runner
========================
Run Apple's Depth Pro model locally on any platform.

Device selection (automatic):
 1. MPS  (Apple Silicon GPU)
 2. CUDA (NVIDIA GPU)
 3. CPU  (fallback)

Usage:
    python run.py

Dependencies:
    pip install -r requirements.txt
"""

import os
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

# ---------------------------------------------------------------------------
# Import depth_pro
# ---------------------------------------------------------------------------
try:
    import depth_pro  # type: ignore
except ImportError:
    print("❌ depth_pro module not found. Did you run `pip install -e .` inside the cloned repo?")
    sys.exit(1)

MODEL_AVAILABLE = hasattr(depth_pro, "create_model_and_transforms")

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🚀 Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA GPU ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("⚠️  No GPU available – running on CPU. Expect slower performance.")

# Globals for lazy model loading
model = None
transform = None
load_time = 0.0


def download_model_if_needed():
    """Download the Depth Pro weights if they don't exist."""
    default_path = Path("depth_pro.pt")
    if default_path.exists():
        return default_path

    url = "https://huggingface.co/apple/depth-pro/resolve/main/depth_pro.pt"
    print("⬇️  depth_pro.pt not found – downloading (1.8 GB)…")
    import urllib.request

    with urllib.request.urlopen(url) as response, open(default_path, "wb") as out_file:
        file_size = int(response.getheader("Content-Length", "0"))
        downloaded = 0
        chunk_size = 8192
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            progress = downloaded / file_size * 100
            print(f"\r   {progress:5.1f}%", end="", flush=True)
    print("\n✅ Model downloaded to depth_pro.pt")
    return default_path


def load_model():
    """Load the Depth Pro model the first time it's needed."""
    global model, transform, load_time
    if model is not None:
        return True

    if not MODEL_AVAILABLE:
        print("❌ depth_pro module does not expose create_model_and_transforms().")
        return False

    weights_path = download_model_if_needed()

    print("🔄 Loading Depth Pro model…")
    start = time.time()
    model, transform = depth_pro.create_model_and_transforms(
        weights_path=str(weights_path)
    )
    model = model.to(device).eval()

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.1f}s on {device}")
    return True


def process_image(image: Image.Image):
    """Run the model on a PIL image and return colour & greyscale depth maps."""
    if image is None:
        return None, None, "❌ Please upload an image first!"

    if not load_model():
        return None, None, "❌ Model failed to load. Check console for details."

    if image.mode != "RGB":
        image = image.convert("RGB")

    start_total = time.time()

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model.infer(tensor)
        else:
            prediction = model.infer(tensor)

    depth = prediction["depth"].cpu().numpy().squeeze()
    focal = prediction["focallength_px"]

    total_time = time.time() - start_total

    # Normalise to 0–255 for visualisation
    d_min, d_max = depth.min(), depth.max()
    depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
    depth_colour = cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB)

    gpu_label = {
        "mps": "Apple Silicon MPS",
        "cuda": f"CUDA ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "",
        "cpu": "None (CPU)",
    }.get(device.type, str(device))

    info_md = f"""
### ✅ Depth Map Generated Successfully!

- **📏 Image Size**: {image.width} × {image.height}
- **🔍 Estimated Focal Length**: {focal:.1f} px
- **📊 Depth Range**: {d_min:.2f} m – {d_max:.2f} m
- **⚡ Processing Time**: {total_time:.3f}s
- **🎯 Model**: Apple Depth Pro v1.0
- **💻 Device**: {device}

**Performance Notes:**
- **Model Load Time**: {load_time:.1f}s (one-time)
- **GPU Acceleration**: {gpu_label}

**How to use the results:**
- **Colored Version**: Great for visualisation and analysis
- **Grayscale Version**: Use for 3D reconstruction, depth-based effects
- **Depth Values**: White = closer, Black = farther
    """
    return depth_colour, depth_norm, info_md


def create_interface():
    """Create the Gradio Blocks interface."""
    css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
        max-width: 1400px; margin: 0 auto;
    }
    .main-header { text-align: center; margin-bottom: 30px; padding: 25px;
        background: linear-gradient(135deg,#007AFF 0%,#5856D6 50%,#AF52DE 100%);
        border-radius: 12px; color: white; box-shadow: 0 4px 20px rgba(0,0,0,.1);
    }
    .performance-badge { display:inline-block; background:rgba(255,255,255,.2);
        padding:8px 16px; border-radius:20px; margin:5px; font-size:14px; }
    .footer { text-align:center; color:#86868b; font-size:.9rem; margin-top:2rem;
        padding-top:1rem; border-top:1px solid #444; }
    """

    with gr.Blocks(title="🎯 Depth Pro - AI Depth Map", css=css) as demo:
        gr.HTML("""
        <div class="main-header">
            <h1>🎯 Depth Pro - AI Depth Map Generator</h1>
            <p style="font-size:20px;margin:15px auto;opacity:0.95;">
                Transform any 2D image into a detailed 3D depth map in <strong>seconds</strong>!<br>
                Powered by <strong>Apple's Depth Pro</strong>.
            </p>
            <div style="margin-top:20px;">
                <span class="performance-badge">🚀 GPU Accelerated</span>
                <span class="performance-badge">🎯 Zero-shot</span>
            </div>
            <p style="margin-top:15px;opacity:0.8;font-size:16px;">
                Running locally on device
            </p>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column():
                gr.HTML("<h3>📤 Upload Your Image</h3>")
                img_in = gr.Image(
                    label="Drop an image or click to upload", type="pil", height=450
                )

                gr.HTML("""
                <div style="margin-top:15px;padding:15px;background:linear-gradient(135deg,#2c2c2e 0%,#1c1c1e 100%);color:#f2f2f7;border:1px solid #444;border-radius:10px;">
                    <strong>💡 Features:</strong>
                    <ul style="margin:10px 0;padding-left:20px;">
                        <li><strong>🚀 Ultra-fast</strong>: GPU acceleration (MPS / CUDA)</li>
                        <li><strong>📸 Any image</strong>: People, objects, landscapes, indoor/outdoor</li>
                        <li><strong>🔥 No limits</strong>: Process unlimited images locally</li>
                        <li><strong>🔒 Private</strong>: All processing happens locally</li>
                    </ul>
                </div>
                """)

            with gr.Column():
                gr.HTML("<h3>🎨 Colored Depth Map</h3>")
                img_out_col = gr.Image(height=450)
            with gr.Column():
                gr.HTML("<h3>⚫ Grayscale Depth Map</h3>")
                img_out_gray = gr.Image(height=450)

        info = gr.Markdown("⌛ Load an image to begin…")

        img_in.change(
            process_image, inputs=img_in, outputs=[img_out_col, img_out_gray, info]
        )

        gr.HTML("""
        <div class="footer">
            🤖 Powered by Apple's Depth Pro | 💻 Running locally on your device<br>
            Research: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (2024)<br>
            🔒 Private processing • 🚀 Local inference
        </div>
        """)

    return demo


def main():
    print("🚀 Starting Depth Pro…")
    load_model()  # warm-up
    demo = create_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
