#!/usr/bin/env python3
"""
Depth Pro - Batch Processor
============================
Process a folder of images and save depth maps to an output directory.

Usage:
    python batch.py --input ./photos --output ./depth_maps

Supports: .jpg, .jpeg, .png, .bmp, .webp, .heif, .heic
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from run import DepthProRunner, _select_device

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heif", ".heic"}


def batch_process(input_dir: Path, output_dir: Path, save_npy: bool = False):
    """Process all images in input_dir, save results to output_dir."""
    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in SUPPORTED
    )

    if not images:
        print(f"❌ No supported images found in {input_dir}")
        print(f"   Supported formats: {', '.join(SUPPORTED)}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device()
    runner = DepthProRunner(device)
    runner.load()

    print(f"\n📂 Processing {len(images)} images → {output_dir}\n")

    total_time = 0.0
    for img_path in tqdm(images, desc="Processing", unit="img"):
        image = Image.open(img_path).convert("RGB")

        start = time.time()
        colour, greyscale, _ = runner.process(image)
        elapsed = time.time() - start
        total_time += elapsed

        stem = img_path.stem

        # Save colored depth map
        colour_bgr = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"{stem}_depth_color.png"), colour_bgr)

        # Save grayscale depth map
        cv2.imwrite(str(output_dir / f"{stem}_depth_gray.png"), greyscale)

        # Optionally save raw depth as NumPy array
        if save_npy:
            np.save(str(output_dir / f"{stem}_depth.npy"), greyscale)

    print(f"\n✅ Done! {len(images)} images processed in {total_time:.1f}s")
    print(f"   Average: {total_time / len(images):.2f}s per image")
    print(f"   Output:  {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch process images with Depth Pro"
    )
    parser.add_argument(
        "--input", "-i", type=Path, required=True,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("depth_output"),
        help="Directory to save depth maps (default: ./depth_output)"
    )
    parser.add_argument(
        "--npy", action="store_true",
        help="Also save raw depth values as .npy files"
    )
    args = parser.parse_args()

    if not args.input.is_dir():
        print(f"❌ Input path is not a directory: {args.input}")
        sys.exit(1)

    batch_process(args.input, args.output, save_npy=args.npy)


if __name__ == "__main__":
    main()
