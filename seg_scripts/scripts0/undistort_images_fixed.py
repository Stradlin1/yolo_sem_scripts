#!/usr/bin/env python3

import os
import glob
import argparse
import cv2
import yaml
import numpy as np


def load_camera_info(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1)

    width = data.get("image_width", None)
    height = data.get("image_height", None)

    return K, D, width, height


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_info", required=True)

    # Kept only for compatibility with your old command.
    # It is NOT used, because BEV homography expects the original-K undistorted image coordinates.
    parser.add_argument("--alpha", type=float, default=0.0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    K, D, calib_w, calib_h = load_camera_info(args.camera_info)

    print("K:")
    print(K)
    print("D:")
    print(D)

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))
    if not image_paths:
        raise RuntimeError(f"No jpg images found in {args.input_dir}")

    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"Cannot read image: {image_paths[0]}")

    h, w = first.shape[:2]

    print("Image size:", w, h)
    if calib_w is not None and calib_h is not None:
        print("Calibration size:", calib_w, calib_h)
        if int(calib_w) != w or int(calib_h) != h:
            print("[WARN] Image size does not match camera_info image_width/image_height.")
            print("[WARN] Homography-based BEV may be wrong unless the calibration/homography is scaled accordingly.")

    # IMPORTANT:
    # Do NOT use cv2.getOptimalNewCameraMatrix here.
    # Keep the original K as the output camera matrix, otherwise the undistorted image coordinate
    # system changes and H_ground_to_undistorted_image may no longer match.
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K, (w, h), cv2.CV_16SC2
    )

    print("Using original K as undistorted output camera matrix.")
    print("Output K:")
    print(K)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] skip {path}")
            continue

        undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, undistorted)

        if i % 50 == 0:
            print(f"[{i}/{len(image_paths)}] {out_path}")

    print("Done.")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
