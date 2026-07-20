#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch raw image -> undistorted image -> vehicle-forward BEV.

This script follows the logic of apply_vehicle_bev.py:

  undistorted image
    -> board/ground frame
    -> vehicle frame
    -> vehicle-forward BEV image

Vehicle frame:
  Xv: forward
  Yv: left

BEV image convention:
  image up    = +Xv forward
  image left  = +Yv left
  image right = -Yv right

Default paths are for:
  /home/xhm/Desktop/relocate_ws
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_yaml(path: Path) -> dict:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def matrix_from_yaml_item(item, rows=None, cols=None) -> np.ndarray:
    if isinstance(item, dict):
        if "data" not in item:
            raise RuntimeError(f"matrix dict has no data field: {item}")
        data = item["data"]
        r = int(item.get("rows", rows))
        c = int(item.get("cols", cols))
        return np.array(data, dtype=np.float64).reshape(r, c)

    arr = np.array(item, dtype=np.float64)
    if rows is not None and cols is not None:
        return arr.reshape(rows, cols)
    return arr


def read_matrix(data: dict, name: str, rows=None, cols=None) -> np.ndarray:
    if name not in data:
        raise RuntimeError(f"missing matrix key: {name}. Available keys: {list(data.keys())}")
    return matrix_from_yaml_item(data[name], rows, cols)


def read_camera_from_ground_or_camera_yaml(ground_data: dict, camera_yaml: Path):
    if "camera_matrix" in ground_data and "distortion_coefficients" in ground_data:
        K = read_matrix(ground_data, "camera_matrix", 3, 3)
        dist = read_matrix(ground_data, "distortion_coefficients").reshape(-1, 1)
        return K, dist, "ground_extrinsic.yaml"

    if not camera_yaml.exists():
        raise RuntimeError(
            "ground_extrinsic.yaml has no camera_matrix/distortion_coefficients, "
            f"and camera yaml does not exist: {camera_yaml}"
        )

    cam_data = read_yaml(camera_yaml)

    if "camera_matrix" not in cam_data:
        raise RuntimeError(f"camera_matrix not found in {camera_yaml}")

    if "distortion_coefficients" not in cam_data:
        raise RuntimeError(f"distortion_coefficients not found in {camera_yaml}")

    K = matrix_from_yaml_item(cam_data["camera_matrix"], 3, 3)
    dist = matrix_from_yaml_item(cam_data["distortion_coefficients"]).reshape(-1, 1)

    return K, dist, str(camera_yaml)


def list_images(input_root: Path, recursive: bool = True) -> List[Path]:
    if input_root.is_file():
        return [input_root]

    it: Iterable[Path] = input_root.rglob("*") if recursive else input_root.iterdir()
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def normalize2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(2)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise RuntimeError("forward vector is too small")
    return v / n


def parse_axis(axis: str) -> np.ndarray:
    axis = axis.strip().lower()

    if axis in {"+x", "x", "board+x", "+board_x"}:
        return np.array([1.0, 0.0], dtype=np.float64)

    if axis in {"-x", "board-x", "-board_x"}:
        return np.array([-1.0, 0.0], dtype=np.float64)

    if axis in {"+y", "y", "board+y", "+board_y"}:
        return np.array([0.0, 1.0], dtype=np.float64)

    if axis in {"-y", "board-y", "-board_y"}:
        return np.array([0.0, -1.0], dtype=np.float64)

    raise RuntimeError("axis must be one of: +x, -x, +y, -y")


def compute_vehicle_basis_in_board(data: dict, args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Returns:
      origin_board_xy: vehicle origin expressed in board/ground XY
      forward_board  : vehicle +X direction expressed in board/ground XY
      left_board     : vehicle +Y direction expressed in board/ground XY
      note
    """

    T_ground_camera = read_matrix(data, "T_ground_camera", 4, 4)
    camera_xy = T_ground_camera[:2, 3].astype(np.float64)

    if args.origin == "camera-ground":
        origin = camera_xy.copy()
    elif args.origin == "board":
        origin = np.array([0.0, 0.0], dtype=np.float64)
    elif args.origin == "manual":
        origin = np.array([args.origin_board_x, args.origin_board_y], dtype=np.float64)
    else:
        raise RuntimeError(f"unknown origin mode: {args.origin}")

    if args.align == "camera-optical":
        R_ground_camera = T_ground_camera[:3, :3]

        # OpenCV camera frame:
        #   +Z is optical axis.
        # Project camera optical axis to ground XY plane as vehicle forward.
        optical_axis_in_ground = R_ground_camera[:, 2]
        forward = normalize2(optical_axis_in_ground[:2])

        note = "vehicle +X is camera optical axis projected to ground"

    elif args.align == "axis":
        forward = normalize2(parse_axis(args.vehicle_forward_axis))
        note = f"vehicle +X is board {args.vehicle_forward_axis}"

    elif args.align == "yaw":
        theta = math.radians(args.yaw_deg)
        forward = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        note = f"vehicle +X yaw in board frame = {args.yaw_deg} deg"

    else:
        raise RuntimeError(f"unknown align mode: {args.align}")

    # Right-handed 2D basis:
    # left is +90 deg rotation from forward.
    left = np.array([-forward[1], forward[0]], dtype=np.float64)

    if args.flip_left:
        left = -left
        note += "; left axis flipped"

    return origin, forward, left, note


def make_T_board_vehicle(origin_board_xy: np.ndarray, forward_board: np.ndarray, left_board: np.ndarray) -> np.ndarray:
    """
    Board XY = origin + forward * X_vehicle + left * Y_vehicle
    """
    T = np.eye(3, dtype=np.float64)
    T[0:2, 0] = forward_board.reshape(2)
    T[0:2, 1] = left_board.reshape(2)
    T[0:2, 2] = origin_board_xy.reshape(2)
    return T


def vehicle_to_bev_matrix(
    forward_min: float,
    forward_max: float,
    left_min: float,
    left_max: float,
    ppm: float,
) -> Tuple[np.ndarray, int, int]:
    """
    Vehicle frame:
      X forward, Y left.

    BEV image:
      u = (left_max - Y) * ppm
      v = (forward_max - X) * ppm

    Meaning:
      +X forward appears upward.
      +Y left appears left.
    """

    out_w = int(round((left_max - left_min) * ppm))
    out_h = int(round((forward_max - forward_min) * ppm))

    if out_w <= 0 or out_h <= 0:
        raise RuntimeError(
            f"invalid BEV size. "
            f"forward range=({forward_min}, {forward_max}), "
            f"left range=({left_min}, {left_max}), ppm={ppm}"
        )

    H_vehicle_to_bev = np.array(
        [
            [0.0, -ppm, left_max * ppm],
            [-ppm, 0.0, forward_max * ppm],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return H_vehicle_to_bev, out_w, out_h


def transform_points_h(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    pts = np.column_stack([pts_xy, np.ones(len(pts_xy), dtype=np.float64)])
    out = (H @ pts.T).T
    out = out[:, :2] / out[:, 2:3]
    return out


def make_vehicle_grid(forward_min, forward_max, left_min, left_max, step):
    lines = []

    xs = np.arange(np.ceil(forward_min / step) * step, forward_max + 1e-9, step)
    ys = np.arange(np.ceil(left_min / step) * step, left_max + 1e-9, step)

    for y in ys:
        lines.append(np.array([[forward_min, y], [forward_max, y]], dtype=np.float64))

    for x in xs:
        lines.append(np.array([[x, left_min], [x, left_max]], dtype=np.float64))

    return lines


def draw_grid_on_undistorted(
    image: np.ndarray,
    H_vehicle_to_img: np.ndarray,
    forward_min: float,
    forward_max: float,
    left_min: float,
    left_max: float,
    step: float,
) -> np.ndarray:
    vis = image.copy()
    lines = make_vehicle_grid(forward_min, forward_max, left_min, left_max, step)

    for line in lines:
        pts = transform_points_h(H_vehicle_to_img, line)
        p0, p1 = np.round(pts).astype(int)
        cv2.line(vis, tuple(p0), tuple(p1), (0, 255, 255), 1)

    origin = transform_points_h(H_vehicle_to_img, np.array([[0.0, 0.0]], dtype=np.float64))[0]
    x_axis = transform_points_h(H_vehicle_to_img, np.array([[0.3, 0.0]], dtype=np.float64))[0]
    y_axis = transform_points_h(H_vehicle_to_img, np.array([[0.0, 0.3]], dtype=np.float64))[0]

    o = tuple(np.round(origin).astype(int))

    cv2.circle(vis, o, 5, (0, 0, 255), -1)
    cv2.arrowedLine(vis, o, tuple(np.round(x_axis).astype(int)), (0, 0, 255), 2)
    cv2.arrowedLine(vis, o, tuple(np.round(y_axis).astype(int)), (0, 255, 0), 2)

    cv2.putText(
        vis,
        "X forward",
        tuple(np.round(x_axis).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        vis,
        "Y left",
        tuple(np.round(y_axis).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )

    return vis


def draw_grid_on_bev(
    bev: np.ndarray,
    forward_min: float,
    forward_max: float,
    left_min: float,
    left_max: float,
    ppm: float,
    step: float,
) -> np.ndarray:
    vis = bev.copy()
    H_vehicle_to_bev, _, _ = vehicle_to_bev_matrix(forward_min, forward_max, left_min, left_max, ppm)
    lines = make_vehicle_grid(forward_min, forward_max, left_min, left_max, step)

    for line in lines:
        pts = transform_points_h(H_vehicle_to_bev, line)
        p0, p1 = np.round(pts).astype(int)
        cv2.line(vis, tuple(p0), tuple(p1), (0, 255, 255), 1)

    return vis


def build_output_paths(out_root: Path, rel_path: Path, args):
    stem = rel_path.stem

    if args.keep_input_extension:
        bev_rel = rel_path
        undist_rel = rel_path
        overlay_rel = rel_path
        bev_grid_rel = rel_path
    else:
        bev_rel = rel_path.with_name(f"{stem}_vehicle_bev.png")
        undist_rel = rel_path.with_name(f"{stem}_undistorted.png")
        overlay_rel = rel_path.with_name(f"{stem}_vehicle_grid.png")
        bev_grid_rel = rel_path.with_name(f"{stem}_vehicle_bev_grid.png")

    bev_path = out_root / "bev" / bev_rel
    undist_path = out_root / "undistorted" / undist_rel
    overlay_path = out_root / "overlay_vehicle_grid" / overlay_rel
    bev_grid_path = out_root / "bev_vehicle_grid" / bev_grid_rel

    return bev_path, undist_path, overlay_path, bev_grid_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch raw images to vehicle-forward BEV using ground_extrinsic.yaml."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/xhm/Desktop/relocate_ws/data/extracted/raw_all_frames"),
        help="raw image root",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/xhm/Desktop/relocate_ws/data/extracted/bev_vehicle_all_frames"),
        help="output root. BEV images will be saved into output-root/bev",
    )

    parser.add_argument(
        "--ground-yaml",
        type=Path,
        default=Path("/home/xhm/Desktop/relocate_ws/config/ground_extrinsic.yaml"),
        help="ground_extrinsic.yaml",
    )

    parser.add_argument(
        "--camera-yaml",
        type=Path,
        default=Path("/home/xhm/Desktop/relocate_ws/config/camera_info_ros.yaml"),
        help="camera_info_ros.yaml. Used only if ground yaml has no camera intrinsics.",
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="recursively read input images",
    )

    parser.add_argument(
        "--align",
        choices=["camera-optical", "axis", "yaw"],
        default="camera-optical",
        help="vehicle forward alignment",
    )

    parser.add_argument(
        "--vehicle-forward-axis",
        default="+y",
        help="used when --align axis. One of: +x, -x, +y, -y",
    )

    parser.add_argument(
        "--yaw-deg",
        type=float,
        default=0.0,
        help="used when --align yaw. Vehicle +X yaw in board frame, deg",
    )

    parser.add_argument(
        "--flip-left",
        action="store_true",
        help="flip vehicle left axis. Use this if BEV is left/right mirrored.",
    )

    parser.add_argument(
        "--origin",
        choices=["camera-ground", "board", "manual"],
        default="camera-ground",
        help="vehicle origin mode",
    )

    parser.add_argument(
        "--origin-board-x",
        type=float,
        default=0.0,
        help="manual origin X in board frame, meter",
    )

    parser.add_argument(
        "--origin-board-y",
        type=float,
        default=0.0,
        help="manual origin Y in board frame, meter",
    )

    parser.add_argument(
        "--forward-min",
        type=float,
        default=0.0,
        help="BEV forward minimum, meter",
    )

    parser.add_argument(
        "--forward-max",
        type=float,
        default=2.0,
        help="BEV forward maximum, meter",
    )

    parser.add_argument(
        "--left-min",
        type=float,
        default=-0.6,
        help="BEV left minimum, meter. Negative means right side.",
    )

    parser.add_argument(
        "--left-max",
        type=float,
        default=0.6,
        help="BEV left maximum, meter. Positive means left side.",
    )

    parser.add_argument(
        "--ppm",
        type=float,
        default=250.0,
        help="pixels per meter",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="cv2.getOptimalNewCameraMatrix alpha. Keep 0 unless you know H was built for another alpha.",
    )

    parser.add_argument(
        "--save-undistorted",
        action="store_true",
        help="save undistorted intermediate images",
    )

    parser.add_argument(
        "--draw-grid",
        action="store_true",
        help="save vehicle grid overlay and BEV grid",
    )

    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="grid step in meters",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="process first N images only. 0 means all",
    )

    parser.add_argument(
        "--keep-input-extension",
        action="store_true",
        help="save outputs with same relative filename and extension as input",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ground_yaml = args.ground_yaml.expanduser().resolve()
    camera_yaml = args.camera_yaml.expanduser().resolve()

    if not input_root.exists():
        raise RuntimeError(f"input root does not exist: {input_root}")

    if not ground_yaml.exists():
        raise RuntimeError(f"ground yaml does not exist: {ground_yaml}")

    ground_data = read_yaml(ground_yaml)

    K, dist, camera_source = read_camera_from_ground_or_camera_yaml(ground_data, camera_yaml)

    H_board_to_img = read_matrix(ground_data, "H_ground_to_undistorted_image", 3, 3)
    H_img_to_board = read_matrix(ground_data, "H_undistorted_image_to_ground", 3, 3)

    origin_board, forward_board, left_board, note = compute_vehicle_basis_in_board(ground_data, args)

    T_board_vehicle = make_T_board_vehicle(origin_board, forward_board, left_board)
    T_vehicle_board = np.linalg.inv(T_board_vehicle)

    H_vehicle_to_img = H_board_to_img @ T_board_vehicle
    H_vehicle_to_img = H_vehicle_to_img / H_vehicle_to_img[2, 2]

    H_img_to_vehicle = T_vehicle_board @ H_img_to_board
    H_img_to_vehicle = H_img_to_vehicle / H_img_to_vehicle[2, 2]

    H_vehicle_to_bev, out_w, out_h = vehicle_to_bev_matrix(
        args.forward_min,
        args.forward_max,
        args.left_min,
        args.left_max,
        args.ppm,
    )

    H_img_to_bev = H_vehicle_to_bev @ H_img_to_vehicle
    H_img_to_bev = H_img_to_bev / H_img_to_bev[2, 2]

    images = list_images(input_root, recursive=True)

    if args.limit > 0:
        images = images[: args.limit]

    if not images:
        raise RuntimeError(f"no images found in: {input_root}")

    bev_dir = output_root / "bev"
    undist_dir = output_root / "undistorted"
    overlay_dir = output_root / "overlay_vehicle_grid"
    bev_grid_dir = output_root / "bev_vehicle_grid"

    bev_dir.mkdir(parents=True, exist_ok=True)

    if args.save_undistorted:
        undist_dir.mkdir(parents=True, exist_ok=True)

    if args.draw_grid:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        bev_grid_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Raw image -> undistorted image -> vehicle-forward BEV")
    print(f"input root        : {input_root}")
    print(f"output root       : {output_root}")
    print(f"ground yaml       : {ground_yaml}")
    print(f"camera source     : {camera_source}")
    print(f"images            : {len(images)}")
    print(f"align             : {args.align}")
    print(f"align note        : {note}")
    print(f"origin mode       : {args.origin}")
    print(f"origin board xy   : {origin_board[0]:.6f}, {origin_board[1]:.6f}")
    print(f"forward in board  : {forward_board[0]:.6f}, {forward_board[1]:.6f}")
    print(f"left in board     : {left_board[0]:.6f}, {left_board[1]:.6f}")
    print(f"forward range     : {args.forward_min:.3f} -> {args.forward_max:.3f} m")
    print(f"left range        : {args.left_min:.3f} -> {args.left_max:.3f} m")
    print(f"ppm               : {args.ppm}")
    print(f"BEV size          : {out_w} x {out_h}")
    print(f"alpha             : {args.alpha}")
    print(f"save undistorted  : {args.save_undistorted}")
    print(f"draw grid         : {args.draw_grid}")
    print("=" * 100)

    rows = []
    processed = 0
    failed = 0
    last_size = None
    current_new_K = None

    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            failed += 1
            rows.append(
                {
                    "filename": str(img_path),
                    "ok": 0,
                    "width": "",
                    "height": "",
                    "bev_width": out_w,
                    "bev_height": out_h,
                    "reason": "imread failed",
                }
            )
            continue

        h, w = img.shape[:2]
        cur_size = (w, h)

        if cur_size != last_size:
            current_new_K, _ = cv2.getOptimalNewCameraMatrix(
                K,
                dist,
                cur_size,
                args.alpha,
                newImgSize=cur_size,
            )
            last_size = cur_size
            print(f"[INFO] init undistort for image size: {w} x {h}")

        undist = cv2.undistort(img, K, dist, None, current_new_K)

        bev = cv2.warpPerspective(
            undist,
            H_img_to_bev,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        rel_path = img_path.relative_to(input_root)

        bev_path, undist_path, overlay_path, bev_grid_path = build_output_paths(output_root, rel_path, args)

        bev_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(bev_path), bev)

        if not ok:
            failed += 1
            rows.append(
                {
                    "filename": str(img_path),
                    "ok": 0,
                    "width": w,
                    "height": h,
                    "bev_width": out_w,
                    "bev_height": out_h,
                    "reason": "failed to write bev",
                }
            )
            continue

        if args.save_undistorted:
            undist_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(undist_path), undist)

        if args.draw_grid:
            overlay = draw_grid_on_undistorted(
                undist,
                H_vehicle_to_img,
                args.forward_min,
                args.forward_max,
                args.left_min,
                args.left_max,
                args.grid_step,
            )

            bev_grid = draw_grid_on_bev(
                bev,
                args.forward_min,
                args.forward_max,
                args.left_min,
                args.left_max,
                args.ppm,
                args.grid_step,
            )

            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            bev_grid_path.parent.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(overlay_path), overlay)
            cv2.imwrite(str(bev_grid_path), bev_grid)

        processed += 1

        rows.append(
            {
                "filename": str(img_path.relative_to(input_root)),
                "ok": 1,
                "width": w,
                "height": h,
                "bev_width": out_w,
                "bev_height": out_h,
                "reason": "ok",
            }
        )

        if processed > 0 and processed % 500 == 0:
            print(f"[INFO] processed {processed}/{len(images)}")

    log_path = output_root / "process_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["filename", "ok", "width", "height", "bev_width", "bev_height", "reason"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})

    print("=" * 100)
    print("DONE")
    print(f"processed : {processed}")
    print(f"failed    : {failed}")
    print(f"BEV dir   : {bev_dir}")
    print(f"log       : {log_path}")

    if args.save_undistorted:
        print(f"undistort : {undist_dir}")

    if args.draw_grid:
        print(f"overlay   : {overlay_dir}")
        print(f"bev grid  : {bev_grid_dir}")

    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
