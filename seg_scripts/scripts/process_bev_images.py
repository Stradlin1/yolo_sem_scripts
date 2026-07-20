#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch image -> vehicle-forward BEV conversion.

This script performs image processing only:
  1. Read raw camera images.
  2. Undistort them using camera_info_ros.yaml.
  3. Convert the undistorted image to a vehicle-forward BEV using
     ground_extrinsic.yaml.
  4. Save only the final BEV images.

Coordinate convention:
  vehicle +X: forward
  vehicle +Y: left
  BEV up:      forward
  BEV left:    vehicle left

Default BEV range:
  forward: 0.00 m .. 1.28 m
  lateral: -0.64 m .. 0.64 m
  resolution: 250 pixels/m
  output: 320 x 320 pixels
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch raw images to 320x320 vehicle-forward BEV images."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input image file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Output BEV directory")
    parser.add_argument(
        "--camera-info",
        type=Path,
        required=True,
        help="ROS-format camera_info_ros.yaml",
    )
    parser.add_argument(
        "--extrinsic",
        type=Path,
        required=True,
        help="ground_extrinsic.yaml",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively and preserve relative subdirectories",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Process every Nth image after sorting; default: 1 (all images)",
    )
    parser.add_argument(
        "--forward-min",
        type=float,
        default=0.0,
        help="Minimum forward distance in metres; default: 0.0",
    )
    parser.add_argument(
        "--forward-max",
        type=float,
        default=1.28,
        help="Maximum forward distance in metres; default: 1.28",
    )
    parser.add_argument(
        "--left-min",
        type=float,
        default=-0.64,
        help="Minimum left coordinate in metres; negative means right; default: -0.64",
    )
    parser.add_argument(
        "--left-max",
        type=float,
        default=0.64,
        help="Maximum left coordinate in metres; default: 0.64",
    )
    parser.add_argument(
        "--ppm",
        type=float,
        default=250.0,
        help="Pixels per metre; default: 250, producing 320x320 with default ranges",
    )
    parser.add_argument(
        "--flip-left",
        action="store_true",
        help="Flip the vehicle left axis if the generated BEV is horizontally mirrored",
    )
    parser.add_argument(
        "--output-ext",
        choices=["png", "jpg"],
        default="png",
        help="Output image format; default: png",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content: {path}")
    return data


def read_matrix(data: dict, key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing matrix '{key}' in YAML")
    item = data[key]
    try:
        rows = int(item["rows"])
        cols = int(item["cols"])
        values = np.asarray(item["data"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid matrix '{key}'") from exc
    if values.size != rows * cols:
        raise ValueError(
            f"Matrix '{key}' has {values.size} values, expected {rows * cols}"
        )
    return values.reshape(rows, cols)


def list_images(input_path: Path, recursive: bool) -> list[Path]:
    input_path = input_path.expanduser().resolve()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def normalize_xy(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Projected camera optical axis is too small")
    return vector / norm


def build_image_to_bev_homography(
    extrinsic: dict,
    forward_min: float,
    forward_max: float,
    left_min: float,
    left_max: float,
    ppm: float,
    flip_left: bool,
) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray, np.ndarray]:
    """Build undistorted-image -> vehicle-forward-BEV homography."""
    if forward_max <= forward_min:
        raise ValueError("forward_max must be greater than forward_min")
    if left_max <= left_min:
        raise ValueError("left_max must be greater than left_min")
    if ppm <= 0:
        raise ValueError("ppm must be positive")

    h_img_to_ground = read_matrix(extrinsic, "H_undistorted_image_to_ground")
    t_ground_camera = read_matrix(extrinsic, "T_ground_camera")

    # Vehicle origin: vertical projection of the camera centre onto the ground plane.
    origin_ground_xy = t_ground_camera[:2, 3].astype(np.float64)

    # OpenCV camera +Z is the optical/forward axis. Project it onto the ground plane.
    optical_axis_ground = t_ground_camera[:3, 2]
    forward_ground = normalize_xy(optical_axis_ground[:2])

    # Standard planar right-handed basis: +Y vehicle points left.
    left_ground = np.array([-forward_ground[1], forward_ground[0]], dtype=np.float64)
    if flip_left:
        left_ground = -left_ground

    # Ground XY = origin + forward * X_vehicle + left * Y_vehicle.
    t_ground_vehicle = np.eye(3, dtype=np.float64)
    t_ground_vehicle[:2, 0] = forward_ground
    t_ground_vehicle[:2, 1] = left_ground
    t_ground_vehicle[:2, 2] = origin_ground_xy
    t_vehicle_ground = np.linalg.inv(t_ground_vehicle)

    # Vehicle metric coordinates -> BEV pixels.
    # u = (left_max - Y_vehicle) * ppm
    # v = (forward_max - X_vehicle) * ppm
    h_vehicle_to_bev = np.array(
        [
            [0.0, -ppm, left_max * ppm],
            [-ppm, 0.0, forward_max * ppm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    h_img_to_vehicle = t_vehicle_ground @ h_img_to_ground
    h_img_to_bev = h_vehicle_to_bev @ h_img_to_vehicle
    if abs(h_img_to_bev[2, 2]) < 1e-12:
        raise ValueError("Invalid image-to-BEV homography")
    h_img_to_bev /= h_img_to_bev[2, 2]

    output_width = int(round((left_max - left_min) * ppm))
    output_height = int(round((forward_max - forward_min) * ppm))
    if output_width <= 0 or output_height <= 0:
        raise ValueError("Computed output size is invalid")

    return (
        h_img_to_bev,
        output_width,
        output_height,
        origin_ground_xy,
        forward_ground,
        left_ground,
    )


def validate_calibration(camera_info: dict, extrinsic: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    model = str(camera_info.get("distortion_model", "plumb_bob"))
    if model not in {"plumb_bob", "rational_polynomial"}:
        raise ValueError(
            f"Unsupported distortion_model '{model}'. This script expects a pinhole camera model."
        )

    camera_width = int(camera_info["image_width"])
    camera_height = int(camera_info["image_height"])
    extrinsic_width = int(extrinsic["image_width"])
    extrinsic_height = int(extrinsic["image_height"])
    if (camera_width, camera_height) != (extrinsic_width, extrinsic_height):
        raise ValueError(
            "camera_info and ground_extrinsic image sizes do not match: "
            f"{camera_width}x{camera_height} vs {extrinsic_width}x{extrinsic_height}"
        )

    k = read_matrix(camera_info, "camera_matrix")
    dist = read_matrix(camera_info, "distortion_coefficients").reshape(-1, 1)

    # Cross-check duplicate intrinsics in ground_extrinsic.yaml.
    k_ext = read_matrix(extrinsic, "camera_matrix")
    dist_ext = read_matrix(extrinsic, "distortion_coefficients").reshape(-1, 1)
    if not np.allclose(k, k_ext, rtol=0.0, atol=1e-6):
        raise ValueError("camera_matrix differs between camera_info and ground_extrinsic")
    if not np.allclose(dist, dist_ext, rtol=0.0, atol=1e-8):
        raise ValueError("distortion_coefficients differ between camera_info and ground_extrinsic")

    return k, dist, camera_width, camera_height


def make_output_path(
    source: Path,
    input_root: Path,
    output_root: Path,
    recursive: bool,
    output_ext: str,
) -> Path:
    if input_root.is_file():
        relative_parent = Path()
    elif recursive:
        relative_parent = source.parent.relative_to(input_root)
    else:
        relative_parent = Path()
    return output_root / relative_parent / f"{source.stem}.{output_ext}"


def main() -> int:
    args = parse_args()
    if args.every_n < 1:
        print("[ERROR] --every-n must be >= 1", file=sys.stderr)
        return 2

    input_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    try:
        camera_info = load_yaml(args.camera_info)
        extrinsic = load_yaml(args.extrinsic)
        k, dist, expected_width, expected_height = validate_calibration(
            camera_info, extrinsic
        )
        (
            h_img_to_bev,
            output_width,
            output_height,
            origin_ground_xy,
            forward_ground,
            left_ground,
        ) = build_image_to_bev_homography(
            extrinsic=extrinsic,
            forward_min=args.forward_min,
            forward_max=args.forward_max,
            left_min=args.left_min,
            left_max=args.left_max,
            ppm=args.ppm,
            flip_left=args.flip_left,
        )
        images = list_images(input_path, args.recursive)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    images = images[:: args.every_n]
    if not images:
        print(f"[ERROR] No input images found: {input_path}", file=sys.stderr)
        return 2

    output_root.mkdir(parents=True, exist_ok=True)

    # ground_extrinsic.yaml stores homographies for alpha=0 undistorted images.
    new_k, _ = cv2.getOptimalNewCameraMatrix(
        k,
        dist,
        (expected_width, expected_height),
        alpha=0.0,
        newImgSize=(expected_width, expected_height),
    )
    map_x, map_y = cv2.initUndistortRectifyMap(
        k,
        dist,
        None,
        new_k,
        (expected_width, expected_height),
        cv2.CV_32FC1,
    )

    print("=" * 72)
    print("Raw image -> vehicle-forward BEV")
    print(f"Input             : {input_path}")
    print(f"Output            : {output_root}")
    print(f"Images selected   : {len(images)}")
    print(f"Expected input    : {expected_width} x {expected_height}")
    print(f"Output BEV        : {output_width} x {output_height}")
    print(
        "Metric range      : "
        f"forward [{args.forward_min:.3f}, {args.forward_max:.3f}] m, "
        f"left [{args.left_min:.3f}, {args.left_max:.3f}] m"
    )
    print(f"Pixels per metre  : {args.ppm:.3f}")
    print(
        "Vehicle origin    : "
        f"ground ({origin_ground_xy[0]:.6f}, {origin_ground_xy[1]:.6f}) m"
    )
    print(
        "Vehicle forward   : "
        f"ground ({forward_ground[0]:.6f}, {forward_ground[1]:.6f})"
    )
    print(
        "Vehicle left      : "
        f"ground ({left_ground[0]:.6f}, {left_ground[1]:.6f})"
    )
    print(f"Horizontal flip   : {'yes' if args.flip_left else 'no'}")
    print("=" * 72)

    processed = 0
    skipped_existing = 0
    failed = 0

    for index, source in enumerate(images, start=1):
        destination = make_output_path(
            source=source,
            input_root=input_path,
            output_root=output_root,
            recursive=args.recursive,
            output_ext=args.output_ext,
        )
        if destination.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Cannot read image: {source}", file=sys.stderr)
            failed += 1
            continue

        height, width = image.shape[:2]
        if (width, height) != (expected_width, expected_height):
            print(
                f"[WARN] Size mismatch, skipped: {source} "
                f"({width}x{height}, expected {expected_width}x{expected_height})",
                file=sys.stderr,
            )
            failed += 1
            continue

        undistorted = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        bev = cv2.warpPerspective(
            undistorted,
            h_img_to_bev,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.output_ext == "jpg":
            ok = cv2.imwrite(str(destination), bev, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            ok = cv2.imwrite(str(destination), bev, [cv2.IMWRITE_PNG_COMPRESSION, 3])

        if not ok:
            print(f"[WARN] Cannot save image: {destination}", file=sys.stderr)
            failed += 1
            continue

        processed += 1
        if index == 1 or index % 100 == 0 or index == len(images):
            print(f"[{index}/{len(images)}] saved: {destination}")

    print("=" * 72)
    print(f"Processed          : {processed}")
    print(f"Skipped existing   : {skipped_existing}")
    print(f"Failed/skipped     : {failed}")
    print(f"BEV directory      : {output_root}")
    print("=" * 72)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
