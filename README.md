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

# 🎯 Depth Pro - AI Depth Map Generator

> **Note:** This repository contains a modified version of the official [Apple/ml-depth-pro](https://github.com/apple/ml-depth-pro) project. The core model and logic are from the original authors. The primary additions are user-friendly local execution scripts (`run_local_mac.py` and `run_local_windows_linux.py`) and an improved README for easier setup. The original license is retained.

---

Transform any 2D image into a detailed 3D depth map using Apple's open-source **Depth Pro** model.

---

## 🚀 Quick Start (Local)

1. **Clone** the repo and enter the directory

    ```bash
    git clone https://github.com/apple/ml-depth-pro.git depth-pro-demo
    cd depth-pro-demo
    ```

2. **Install dependencies** (Python ≥ 3.10 recommended)

    ```bash
    pip install -r requirements.txt
    # For Apple Silicon you also need a recent PyTorch build with MPS support
    ```

3. **Download model weights** (≈ 1.8 GB)

    The helper scripts will automatically download `depth_pro.pt` on first run, but you can also do it manually:

    ```bash
    curl -L -o depth_pro.pt https://huggingface.co/apple/depth-pro/resolve/main/depth_pro.pt
    ```

4. **Run the demo**

    | Platform                     | Command                              |
    |------------------------------|--------------------------------------|
    | macOS (Apple Silicon)        | `python run_local_mac.py`            |
    | Windows or Linux (CUDA/CPU)  | `python run_local_windows_linux.py`  |

    The server starts at <http://127.0.0.1:7860> and opens in your browser automatically.

---

## ✨ Features

- 🚀 **GPU accelerated** (MPS or CUDA) – CPU fallback available
- 🎯 **Metric depth** with absolute scale & focal-length estimation
- 🔒 **Private** – everything runs locally; no data leaves your machine
- 🆓 **Open-source** – no API keys or paywalls

---

## ℹ️  Scripts

| File                           | Description                                                         |
|--------------------------------|---------------------------------------------------------------------|
| `run_local_mac.py`             | Optimised for Apple Silicon (MPS). Works on any M-series Mac.        |
| `run_local_windows_linux.py`   | Works on Windows & Linux. Uses CUDA if available, otherwise CPU.     |

Each script:

1. Checks for **GPU support** (MPS or CUDA) and selects the best device.
2. Downloads `depth_pro.pt` automatically if it’s missing.
3. Launches a **Gradio** interface with coloured & grayscale depth maps.

---

## 📖 Research

Based on the paper:

> **Depth Pro: Sharp Monocular Metric Depth in Less Than a Second** (2024)  
> A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. Richter, and V. Koltun.

---

## 🤝 Credits

- **Model:** [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)  
- **Interface:** [Gradio](https://gradio.app/)  
- **Icons:** [Twemoji](https://twemoji.twitter.com/)

---

*Made with ❤️  for the developer community – enjoy!*
