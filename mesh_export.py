"""
Depth Pro - 3D Mesh Exporter
=============================
Convert depth maps to 3D meshes (.obj, .glb) for AR viewing.
"""

import numpy as np
from PIL import Image


class MeshExporter:
    """Converts a depth map + image into a textured 3D mesh."""

    def __init__(self, depth: np.ndarray, image: Image.Image, focal_px: float):
        """
        Args:
            depth: 2D array of metric depth values (H x W)
            image: original RGB image (will be resized to match depth)
            focal_px: estimated focal length in pixels
        """
        self.depth = depth
        self.h, self.w = depth.shape
        self.image = image.resize((self.w, self.h))
        self.focal = focal_px

        self._vertices = None
        self._faces = None
        self._vertex_colors = None

    # ------------------------------------------------------------------
    # Vertex generation
    # ------------------------------------------------------------------
    def _build_vertices(self, stride: int = 1) -> np.ndarray:
        """Back-project depth pixels into 3D space.

        Args:
            stride: sample every Nth pixel (higher = fewer vertices, faster)

        Returns:
            (N, 3) array of XYZ coordinates
        """
        ys = np.arange(0, self.h, stride)
        xs = np.arange(0, self.w, stride)
        u, v = np.meshgrid(xs, ys)

        cx, cy = self.w / 2.0, self.h / 2.0
        z = self.depth[v, u]
        x = (u - cx) * z / self.focal
        y = (v - cy) * z / self.focal

        # Stack into (N, 3), flip Y for right-hand coordinate system
        vertices = np.stack([x, -y, -z], axis=-1)
        return vertices, u, v
