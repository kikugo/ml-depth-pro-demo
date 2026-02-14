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
