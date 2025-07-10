# Depth Pro - AI Depth Map Generator

Transform any 2D image into a detailed 3D depth map using Apple's open-source **Depth Pro** model.

---

This project provides two easy ways to get started: running simple Python scripts or using an interactive Jupyter Notebook.

## Option 1: Run with Python Scripts (Local)

This is the most straightforward way to run the model on your local machine.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/apple/ml-depth-pro.git
    cd ml-depth-pro
    ```

2.  **Install dependencies** (Python ≥ 3.10 is recommended):
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For Apple Silicon, you will need a PyTorch version that supports MPS.*

3.  **Download Model Weights** (approx. 1.8 GB):
    The scripts will automatically download `depth_pro.pt` on the first run.

4.  **Run the Demo**:
    - For **Mac**: `python run_local_mac.py`
    - For **Windows/Linux**: `python run_local_windows_linux.py`

    This will launch a local Gradio interface in your browser to process images.

---

## Option 2: Run with Jupyter Notebook (Universal)

Use the `Depth_Pro_Universal.ipynb` notebook for an interactive experience that works everywhere.

-   **How it Works**: The notebook uses `ipywidgets` for a native file-upload button. After you upload an image and click "Generate," it displays the original image, colored depth map, and grayscale depth map directly in the notebook output, along with performance metrics.
-   **Where to Run**:
    -   **Google Colab**: Upload the notebook, and it will set up the environment for you.
    -   **Local Jupyter**: Run `jupyter notebook Depth_Pro_Universal.ipynb` in the project directory.
---

## Features

- **GPU Accelerated**: Optimized for Apple Silicon (MPS) and NVIDIA (CUDA), with a CPU fallback.
- **Metric Depth**: Predicts true, real-world depth and estimates focal length automatically.
- **Private**: All processing happens 100% locally on your machine.
- **Two Ways to Run**: Choose between simple command-line scripts or an all-in-one Jupyter Notebook.
- **Open-Source**: Based on Apple's powerful research, free to use without API keys.

---

## Research

This work is based on the paper "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" (2024) by A. Bochkovskii, A. Delaunoy, H. Germain, M. Santos, Y. Zhou, S. Richter, and V. Koltun.

---

> **Note:** This is a personal project to try out Apple's [ml-depth-pro](https://github.com/apple/ml-depth-pro) model. After getting it working locally with easy-to-use scripts, I decided to publish it so anyone else interested can try it too. The core model, logic, and license are from the original Apple repository.


## Credits

- **Original Model**: [apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
- **UI Framework (for scripts)**: [Gradio](https://gradio.app/)
- **Notebook Widgets**: [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)
