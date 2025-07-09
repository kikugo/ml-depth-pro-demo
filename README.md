---
title: Depth Pro - AI Depth Maps
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: other
---

# Depth Pro - AI Depth Map Generator

> **Note:** This is a personal project to try out Apple's ml-depth-pro model. After getting it working locally with easy-to-use scripts, I decided to publish it so anyone else interested can try it too. The core model and logic are from the original Apple repository. The original license is retained.

---

Transform any 2D image into a detailed 3D depth map using Apple's open-source **Depth Pro** model.

---

## Quick Start (Local)

1.  **Clone the repository** and enter the directory:

    ```bash
    git clone https://github.com/apple/ml-depth-pro.git depth-pro-demo
    cd depth-pro-demo
    ```

2.  **Install dependencies** (Python ≥ 3.10 is recommended):

    ```bash
    pip install -r requirements.txt
    ```
    *Note: For Apple Silicon, you will need a PyTorch version that supports MPS.*

3.  **Download Model Weights** (approx. 1.8 GB):

    The scripts will automatically download the `depth_pro.pt` file on the first run. You can also download it manually:

    ```bash
    curl -L -o depth_pro.pt https://huggingface.co/apple/depth-pro/resolve/main/depth_pro.pt
    ```

4.  **Run the Demo**:

    For Mac: `python run_local_mac.py`
    
    For Windows or Linux: `python run_local_windows_linux.py`

    The application will start at http://127.0.0.1:7860 and open in your browser automatically.

---

## Features

- **GPU Accelerated**: Optimized for Apple Silicon (MPS) and NVIDIA (CUDA), with a CPU fallback.
- **Metric Depth**: Predicts true, real-world depth and estimates focal length automatically.
- **Private**: All processing happens 100% locally on your machine. No data is sent to the cloud.
- **Open-Source**: Based on Apple's powerful research, free to use without API keys or restrictions.

---

## Research

This work is based on the paper "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (2024) by A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. Richter, and V. Koltun.

---

## Credits

- **Original Model**: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
- **UI Framework**: [Gradio](https://gradio.app/)
