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

    # ------------------------------------------------------------------
    # Face generation
    # ------------------------------------------------------------------
    def _build_faces(self, rows: int, cols: int, depth_grid: np.ndarray,
                     max_depth_jump: float = 0.3) -> np.ndarray:
        """Create triangle faces for a grid, skipping depth discontinuities.

        Each grid cell becomes two triangles:
            v0 -- v1        tri1: (v0, v3, v1)
            |   / |         tri2: (v0, v2, v3)
            v2 -- v3

        Faces where any edge spans a depth jump > max_depth_jump * median
        are dropped to avoid stretchy artefacts at object boundaries.

        Returns:
            (F, 3) array of vertex indices
        """
        faces = []
        median_depth = np.median(depth_grid[depth_grid > 0]) if np.any(depth_grid > 0) else 1.0
        threshold = max_depth_jump * median_depth

        for r in range(rows - 1):
            for c in range(cols - 1):
                v0 = r * cols + c
                v1 = r * cols + (c + 1)
                v2 = (r + 1) * cols + c
                v3 = (r + 1) * cols + (c + 1)

                # Depth values at the four corners
                d = depth_grid[r, c], depth_grid[r, c + 1]
                d2 = depth_grid[r + 1, c], depth_grid[r + 1, c + 1]

                # Skip if any edge has a large depth jump
                corners = [d[0], d[1], d2[0], d2[1]]
                if max(corners) - min(corners) > threshold:
                    continue

                faces.append([v0, v2, v3])  # lower-left triangle
                faces.append([v0, v3, v1])  # upper-right triangle

        return np.array(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)

    # ------------------------------------------------------------------
    # Color mapping
    # ------------------------------------------------------------------
    def _build_vertex_colors(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Sample the original image at each vertex position.

        Returns:
            (N, 4) array of RGBA values (0-255)
        """
        img_arr = np.array(self.image)
        r = img_arr[v, u, 0].ravel()
        g = img_arr[v, u, 1].ravel()
        b = img_arr[v, u, 2].ravel()
        a = np.full_like(r, 255)
        return np.stack([r, g, b, a], axis=-1).astype(np.uint8)
