#!/usr/bin/env python3
"""
Depth Pro - Local Runner (Windows/Linux)
========================================
Run Apple's Depth Pro model locally on Windows or Linux.

Device selection order:
 1. CUDA GPU (if available)
 2. CPU fallback

Usage:
    python run_local_windows_linux.py

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

try:
    import depth_pro  # type: ignore
except ImportError:
    print("❌ depth_pro module not found. Did you run `pip install -e .` inside the cloned repo?")
    sys.exit(1)

MODEL_AVAILABLE = hasattr(depth_pro, "create_model_and_transforms")

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 Using CUDA GPU ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("⚠️  CUDA not available – running on CPU. Expect slower performance.")

model = None
transform = None
load_time = 0.0


def download_model_if_needed():
    """Download model weights if missing."""
    weights = Path("depth_pro.pt")
    if weights.exists():
        return weights
    url = "https://huggingface.co/apple/depth-pro/resolve/main/depth_pro.pt"
    print("⬇️  depth_pro.pt not found – downloading (1.8 GB)…")
    import urllib.request
    with urllib.request.urlopen(url) as resp, open(weights, "wb") as f:
        size = int(resp.getheader("Content-Length", "0"))
        done = 0
        chunk = 8192
        while True:
            data = resp.read(chunk)
            if not data:
                break
            f.write(data)
            done += len(data)
            print(f"\r   {done/size*100:5.1f}%", end="", flush=True)
    print("\n✅ Model downloaded.")
    return weights


def load_model():
    global model, transform, load_time
    if model is not None:
        return True
    if not MODEL_AVAILABLE:
        return False

    weights_path = download_model_if_needed()

    print("🔄 Loading Depth Pro model…")
    start = time.time()
    model, transform = depth_pro.create_model_and_transforms(weights_path=str(weights_path))
    model = model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    load_time = time.time() - start
    print(f"✅ Model ready in {load_time:.1f}s on {device}")
    return True


def process_image(image: Image.Image):
    if image is None:
        return None, None, "❌ Please upload an image first!"
    if not load_model():
        return None, None, "❌ Model failed to load."
    if image.mode != "RGB":
        image = image.convert("RGB")

    start = time.time()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                pred = model.infer(tensor)
        else:
            pred = model.infer(tensor)

    depth = pred["depth"].cpu().numpy().squeeze()
    focal = pred["focallength_px"]
    elapsed = time.time() - start

    d_min, d_max = depth.min(), depth.max()
    depth_u8 = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    depth_col = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    depth_col = cv2.cvtColor(depth_col, cv2.COLOR_BGR2RGB)

    info = f"""
### ✅ Depth Map Generated Successfully!

- **📏 Image Size**: {image.width} × {image.height}
- **🔍 Estimated Focal Length**: {focal:.1f} px  
- **📊 Depth Range**: {d_min:.2f} m – {d_max:.2f} m
- **⚡ Processing Time**: {elapsed:.3f}s
- **🎯 Model**: Apple Depth Pro v1.0
- **💻 Device**: {device}

**Performance Notes:**
- **Model Load Time**: {load_time:.1f}s (one-time)
- **GPU Acceleration**: {'✅' if device.type == 'cuda' else '❌'}
    """
    return depth_col, depth_u8, info


def ui():
    css = """
    .gradio-container{max-width:1400px;margin:0 auto;}
    .footer{text-align:center;color:#86868b;font-size:.9rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #444;}
    """
    with gr.Blocks(title="🎯 Depth Pro - AI Depth Map (Windows/Linux)", css=css) as demo:
        gr.Markdown("## 🎯 Depth Pro - AI Depth Map Generator\nUpload an image to get started.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Image", height=450)
            with gr.Column():
                out_col = gr.Image(label="Colored Depth", height=450)
            with gr.Column():
                out_gray = gr.Image(label="Grayscale Depth", height=450)
        info = gr.Markdown()
        inp.change(process_image, inputs=inp, outputs=[out_col, out_gray, info])
        gr.HTML("""
        <div class="footer">
            🤖 Powered by Apple's Depth Pro | 💻 Running locally on your device<br>
            Research: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (2024)<br>
            Performance: CUDA acceleration • Zero network latency<br>
            🔒 Private processing • 🚀 Local inference
        </div>
        """)
    return demo


def main():
    print("🚀 Starting Depth Pro (Windows/Linux)…")
    demo = ui()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main() 