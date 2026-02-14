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

import sys
import tempfile
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


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("🚀 Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"🚀 Using CUDA GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    print("⚠️  No GPU available – running on CPU. Expect slower performance.")
    return torch.device("cpu")


class DepthProRunner:
    """Manages model lifecycle and inference."""

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.transform = None
        self.load_time = 0.0

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    @staticmethod
    def download_weights() -> Path:
        """Download the Depth Pro weights if they don't exist."""
        weights = Path("depth_pro.pt")
        if weights.exists():
            return weights

        url = "https://huggingface.co/apple/depth-pro/resolve/main/depth_pro.pt"
        print("⬇️  depth_pro.pt not found – downloading (1.8 GB)…")
        import urllib.request
        from tqdm import tqdm

        with urllib.request.urlopen(url) as resp, open(weights, "wb") as f:
            total = int(resp.getheader("Content-Length", "0"))
            with tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    pbar.update(len(chunk))
        print("✅ Model downloaded to depth_pro.pt")
        return weights

    def load(self) -> bool:
        """Load the model (lazy, only runs once)."""
        if self.model is not None:
            return True

        if not hasattr(depth_pro, "create_model_and_transforms"):
            print("❌ depth_pro module does not expose create_model_and_transforms().")
            return False

        weights_path = self.download_weights()

        print("🔄 Loading Depth Pro model…")
        start = time.time()
        self.model, self.transform = depth_pro.create_model_and_transforms(
            weights_path=str(weights_path)
        )
        self.model = self.model.to(self.device).eval()

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.load_time = time.time() - start
        print(f"✅ Model loaded in {self.load_time:.1f}s on {self.device}")
        return True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def process(self, image: Image.Image):
        """Run inference and return (colour_map, greyscale_map, info_markdown)."""
        if image is None:
            return None, None, "❌ Please upload an image first!"

        if not self.load():
            return None, None, "❌ Model failed to load. Check console for details."

        if image.mode != "RGB":
            image = image.convert("RGB")

        start = time.time()
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    prediction = self.model.infer(tensor)
            else:
                prediction = self.model.infer(tensor)

        depth = prediction["depth"].cpu().numpy().squeeze()
        focal = prediction["focallength_px"]
        elapsed = time.time() - start

        # Normalise to 0–255 for visualisation
        d_min, d_max = depth.min(), depth.max()
        depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        depth_colour = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
        depth_colour = cv2.cvtColor(depth_colour, cv2.COLOR_BGR2RGB)

        gpu_label = {
            "mps": "Apple Silicon MPS",
            "cuda": (
                f"CUDA ({torch.cuda.get_device_name(0)})"
                if self.device.type == "cuda"
                else ""
            ),
            "cpu": "None (CPU)",
        }.get(self.device.type, str(self.device))

        info = f"""
### ✅ Depth Map Generated Successfully!

- **📏 Image Size**: {image.width} × {image.height}
- **🔍 Estimated Focal Length**: {focal:.1f} px
- **📊 Depth Range**: {d_min:.2f} m – {d_max:.2f} m
- **⚡ Processing Time**: {elapsed:.3f}s
- **🎯 Model**: Apple Depth Pro v1.0
- **💻 Device**: {self.device}

**Performance Notes:**
- **Model Load Time**: {self.load_time:.1f}s (one-time)
- **GPU Acceleration**: {gpu_label}

**How to use the results:**
- **Colored Version**: Great for visualisation and analysis
- **Grayscale Version**: Use for 3D reconstruction, depth-based effects
- **Depth Values**: White = closer, Black = farther
        """

        # Save raw depth to a temp .npy for download
        npy_path = Path(tempfile.gettempdir()) / "depth_pro_latest.npy"
        np.save(str(npy_path), depth)

        return depth_colour, depth_norm, info, str(npy_path)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def create_interface(runner: DepthProRunner):
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

    with gr.Blocks(
        title="🎯 Depth Pro - AI Depth Map",
        css=css,
        theme=gr.themes.Soft(),
    ) as demo:
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

        with gr.Tabs():
            # ---- Tab 1: Depth Map -----------------------------------------
            with gr.Tab("🖼️ Depth Map"):
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.HTML("<h3>📤 Upload Your Image</h3>")
                        img_in = gr.Image(
                            label="Drop an image or click to upload",
                            type="pil", height=450,
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

                with gr.Row():
                    info = gr.Markdown("⌛ Load an image to begin…")
                    npy_download = gr.File(
                        label="📥 Download raw depth (.npy)", visible=False
                    )

                def _on_image(image):
                    result = runner.process(image)
                    if result[0] is None:
                        return result[0], result[1], result[2], gr.update(visible=False)
                    return result[0], result[1], result[2], gr.update(
                        value=result[3], visible=True
                    )

                img_in.change(
                    _on_image, inputs=img_in,
                    outputs=[img_out_col, img_out_gray, info, npy_download]
                )

        gr.HTML("""
        <div class="footer">
            🤖 Powered by Apple's Depth Pro | 💻 Running locally on your device<br>
            Research: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (2024)<br>
            🔒 Private processing • 🚀 Local inference
        </div>
        """)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("🚀 Starting Depth Pro…")
    device = _select_device()
    runner = DepthProRunner(device)
    runner.load()  # warm-up
    demo = create_interface(runner)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True)


if __name__ == "__main__":
    main()
