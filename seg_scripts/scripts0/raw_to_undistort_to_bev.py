#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_camera_calib(camera_yaml):
    data = read_yaml(camera_yaml)

    # ROS camera_info.yaml 格式
    if "camera_matrix" in data:
        K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    elif "K" in data:
        K = np.array(data["K"], dtype=np.float64).reshape(3, 3)
    else:
        raise RuntimeError("Cannot find camera_matrix or K in camera yaml")

    if "distortion_coefficients" in data:
        D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)
    elif "D" in data:
        D = np.array(data["D"], dtype=np.float64).reshape(-1)
    else:
        raise RuntimeError("Cannot find distortion_coefficients or D in camera yaml")

    width = None
    height = None

    if "image_width" in data:
        width = int(data["image_width"])
    elif "width" in data:
        width = int(data["width"])

    if "image_height" in data:
        height = int(data["image_height"])
    elif "height" in data:
        height = int(data["height"])

    return K, D, width, height


def get_nested(data, names):
    for name in names:
        if name in data:
            return data[name]
    return None


def load_homography(bev_yaml):
    data = read_yaml(bev_yaml)

    # 兼容常见命名
    candidates = [
        "H",
        "homography",
        "H_img_to_bev",
        "H_image_to_bev",
        "image_to_bev",
        "M",
        "perspective_matrix",
        "warp_matrix",
    ]

    H_data = get_nested(data, candidates)

    if H_data is None:
        # 有些 yaml 会写成:
        # homography:
        #   data: [...]
        for key in candidates:
            if key in data and isinstance(data[key], dict) and "data" in data[key]:
                H_data = data[key]["data"]
                break

    if H_data is None:
        print("Available keys in bev yaml:")
        for k in data.keys():
            print(f"  {k}")
        raise RuntimeError("Cannot find homography matrix in bev yaml")

    if isinstance(H_data, dict) and "data" in H_data:
        H_data = H_data["data"]

    H = np.array(H_data, dtype=np.float64).reshape(3, 3)
    return H


def resolve_bev_size(bev_yaml, default_size):
    data = read_yaml(bev_yaml)

    w = None
    h = None

    for key in ["bev_width", "output_width", "width", "w"]:
        if key in data:
            w = int(data[key])
            break

    for key in ["bev_height", "output_height", "height", "h"]:
        if key in data:
            h = int(data[key])
            break

    if w is None or h is None:
        return default_size

    return (w, h)


def iter_images(input_root):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in input_root.rglob("*") if p.suffix.lower() in exts])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        default="/home/xhm/Desktop/relocate_ws/data/extracted/raw_all_frames",
        help="raw image root folder",
    )

    parser.add_argument(
        "--undistort-root",
        default="/home/xhm/Desktop/relocate_ws/data/extracted/undistorted_all_frames",
        help="undistorted image output root",
    )

    parser.add_argument(
        "--bev-root",
        default="/home/xhm/Desktop/relocate_ws/data/extracted/bev_vehicle_all_frames",
        help="BEV image output root",
    )

    parser.add_argument(
        "--camera-yaml",
        default="/home/xhm/Desktop/relocate_ws/config/camera_info_ros.yaml",
        help="camera intrinsic yaml",
    )

    parser.add_argument(
        "--bev-yaml",
        default="/home/xhm/Desktop/relocate_ws/config/vehicle_bev_extrinsic.yaml",
        help="vehicle BEV extrinsic / homography yaml",
    )

    parser.add_argument(
        "--bev-width",
        type=int,
        default=320,
        help="BEV output width",
    )

    parser.add_argument(
        "--bev-height",
        type=int,
        default=320,
        help="BEV output height",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="cv2.getOptimalNewCameraMatrix alpha, 0 keeps valid pixels, 1 keeps full FOV",
    )

    parser.add_argument(
        "--save-undistort",
        action="store_true",
        help="also save undistorted images",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    undistort_root = Path(args.undistort_root).expanduser().resolve()
    bev_root = Path(args.bev_root).expanduser().resolve()

    camera_yaml = Path(args.camera_yaml).expanduser().resolve()
    bev_yaml = Path(args.bev_yaml).expanduser().resolve()

    if not input_root.exists():
        raise RuntimeError(f"input root does not exist: {input_root}")
    if not camera_yaml.exists():
        raise RuntimeError(f"camera yaml does not exist: {camera_yaml}")
    if not bev_yaml.exists():
        raise RuntimeError(f"bev yaml does not exist: {bev_yaml}")

    K, D, calib_w, calib_h = load_camera_calib(camera_yaml)
    H_img_to_bev = load_homography(bev_yaml)

    bev_size = resolve_bev_size(bev_yaml, (args.bev_width, args.bev_height))

    image_paths = iter_images(input_root)

    if not image_paths:
        raise RuntimeError(f"No images found in: {input_root}")

    bev_root.mkdir(parents=True, exist_ok=True)
    if args.save_undistort:
        undistort_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Raw -> undistort -> BEV")
    print(f"input root      : {input_root}")
    print(f"undistort root  : {undistort_root}")
    print(f"bev root        : {bev_root}")
    print(f"camera yaml     : {camera_yaml}")
    print(f"bev yaml        : {bev_yaml}")
    print(f"images          : {len(image_paths)}")
    print(f"bev size        : {bev_size[0]} x {bev_size[1]}")
    print(f"save undistort  : {args.save_undistort}")
    print("=" * 80)

    map1 = None
    map2 = None
    last_size = None

    count = 0
    failed = 0

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            failed += 1
            print(f"[WARN] failed to read: {img_path}")
            continue

        h, w = img.shape[:2]
        current_size = (w, h)

        if last_size != current_size:
            new_K, _ = cv2.getOptimalNewCameraMatrix(
                K,
                D,
                current_size,
                args.alpha,
                current_size,
            )

            map1, map2 = cv2.initUndistortRectifyMap(
                K,
                D,
                None,
                new_K,
                current_size,
                cv2.CV_16SC2,
            )

            last_size = current_size

            print(f"[INFO] image size changed or initialized: {w} x {h}")

        undistorted = cv2.remap(
            img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        rel_path = img_path.relative_to(input_root)

        if args.save_undistort:
            undistort_path = undistort_root / rel_path
            undistort_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(undistort_path), undistorted)

        bev = cv2.warpPerspective(
            undistorted,
            H_img_to_bev,
            bev_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        bev_path = bev_root / rel_path
        bev_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(bev_path), bev)

        count += 1

        if count % 500 == 0:
            print(f"[INFO] processed {count}/{len(image_paths)}")

    print("=" * 80)
    print(f"DONE")
    print(f"processed : {count}")
    print(f"failed    : {failed}")
    print(f"bev root  : {bev_root}")
    if args.save_undistort:
        print(f"undistort : {undistort_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()