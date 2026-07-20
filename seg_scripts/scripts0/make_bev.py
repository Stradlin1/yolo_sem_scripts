#!/usr/bin/env python3

import os
import glob
import argparse
import cv2
import yaml
import numpy as np


def load_matrix_from_yaml(data, key, shape):
    item = data[key]
    return np.array(item["data"], dtype=np.float64).reshape(shape)


def load_ground_extrinsic(path):
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    H_ground_to_img = load_matrix_from_yaml(
        data, "H_ground_to_undistorted_image", (3, 3)
    )

    return H_ground_to_img


def build_bev_homography(H_ground_to_img, x_min, x_max, y_min, y_max, out_w, out_h):
    """
    ground coordinate:
      x, y in meters on ground plane

    BEV pixel coordinate:
      u from left to right
      v from top to bottom

    Here:
      u = (y - y_min) / (y_max - y_min) * out_w
      v = (x_max - x) / (x_max - x_min) * out_h

    This makes larger x appear toward the top of BEV.
    """

    ground_pts = np.array([
        [x_max, y_min],  # top-left in BEV
        [x_max, y_max],  # top-right
        [x_min, y_max],  # bottom-right
        [x_min, y_min],  # bottom-left
    ], dtype=np.float32)

    ground_pts_h = np.concatenate(
        [ground_pts, np.ones((4, 1), dtype=np.float32)],
        axis=1
    )

    img_pts_h = (H_ground_to_img @ ground_pts_h.T).T
    img_pts = img_pts_h[:, :2] / img_pts_h[:, 2:3]
    img_pts = img_pts.astype(np.float32)

    bev_pts = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    H_img_to_bev = cv2.getPerspectiveTransform(img_pts, bev_pts)

    return H_img_to_bev, img_pts, bev_pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ground_extrinsic", required=True)

    parser.add_argument("--out_w", type=int, default=320)
    parser.add_argument("--out_h", type=int, default=320)

    # First try. You may need to adjust these after looking at the result.
    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=1.2)
    parser.add_argument("--y_min", type=float, default=-0.6)
    parser.add_argument("--y_max", type=float, default=0.6)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    H_ground_to_img = load_ground_extrinsic(args.ground_extrinsic)

    H_img_to_bev, img_pts, bev_pts = build_bev_homography(
        H_ground_to_img=H_ground_to_img,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        out_w=args.out_w,
        out_h=args.out_h,
    )

    print("H_ground_to_undistorted_image:")
    print(H_ground_to_img)
    print("Selected image points:")
    print(img_pts)
    print("BEV points:")
    print(bev_pts)
    print("H_img_to_bev:")
    print(H_img_to_bev)

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))
    if not image_paths:
        raise RuntimeError(f"No jpg images found in {args.input_dir}")

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] skip {path}")
            continue

        bev = cv2.warpPerspective(
            img,
            H_img_to_bev,
            (args.out_w, args.out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        out_path = os.path.join(args.output_dir, os.path.basename(path))
        cv2.imwrite(out_path, bev)

        if i % 50 == 0:
            print(f"[{i}/{len(image_paths)}] {out_path}")

    print("Done.")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()