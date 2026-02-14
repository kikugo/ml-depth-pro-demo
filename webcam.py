#!/usr/bin/env python3
"""
Depth Pro - Live Webcam Mode
==============================
Real-time depth estimation from a webcam feed.

Usage:
    python webcam.py                      # default camera
    python webcam.py --camera 1           # select camera index
    python webcam.py --width 640 --height 480

Controls (OpenCV window):
    q / ESC   — quit
    SPACE     — pause / resume
    c         — cycle colormap
    s         — save screenshot
    r         — start / stop recording

Requirements:
    Works best with CUDA GPU. MPS and CPU will have lower FPS.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

# Available colormaps to cycle through
COLORMAPS = [
    ("Plasma", cv2.COLORMAP_PLASMA),
    ("Inferno", cv2.COLORMAP_INFERNO),
    ("Magma", cv2.COLORMAP_MAGMA),
    ("Viridis", cv2.COLORMAP_VIRIDIS),
    ("Turbo", cv2.COLORMAP_TURBO),
    ("Jet", cv2.COLORMAP_JET),
]


class WebcamDepthRunner:
    """Real-time depth estimation from a webcam feed."""

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480):
        self.camera_index = camera_index
        self.req_width = width
        self.req_height = height

        # State
        self.cap = None
        self.runner = None  # DepthProRunner instance
        self.paused = False
        self.recording = False
        self.writer = None
        self.colormap_idx = 0

        # FPS tracking
        self._frame_times = []
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_camera(self) -> bool:
        """Open the webcam and configure resolution."""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"❌ Could not open camera {self.camera_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cam_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        print(f"📷 Camera opened: {actual_w}×{actual_h} @ {cam_fps:.0f}fps")
        return True

    def _init_model(self) -> bool:
        """Load the Depth Pro model via DepthProRunner."""
        from run import DepthProRunner, _select_device

        device = _select_device()
        self.runner = DepthProRunner(device)
        return self.runner.load()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _infer_frame(self, frame_bgr: np.ndarray):
        """Run depth estimation on a single frame."""
        # Convert to PIL RGB for the model
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        pil_img.thumbnail((1024, 1024))  # Limit max resolution for speed

        # Run inference using the runner's transform and model directly for speed
        # (skipping the full process() wrapper to avoid overhead)
        start = time.time()
        tensor = self.runner.transform(pil_img).unsqueeze(0).to(self.runner.device)

        with torch.no_grad():
            if self.runner.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = self.runner.model.infer(tensor)
            else:
                pred = self.runner.model.infer(tensor)

        depth = pred["depth"].cpu().numpy().squeeze()
        inference_time = time.time() - start

        # Update FPS
        self._frame_times.append(time.time())
        if len(self._frame_times) > 10:
            self._frame_times.pop(0)
        if len(self._frame_times) > 1:
            self._fps = len(self._frame_times) / (
                self._frame_times[-1] - self._frame_times[0]
            )

        return depth, inference_time

