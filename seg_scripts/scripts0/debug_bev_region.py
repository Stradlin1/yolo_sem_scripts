#!/usr/bin/env python3

import os
import cv2
import yaml
import argparse
import numpy as np


def load_matrix(data, key):
    item = data[key]
    return np.array(item["data"], dtype=np.float64).reshape(3, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--ground_extrinsic", required=True)
    parser.add_argument("--out", required=True)

    parser.add_argument("--x_min", type=float, default=0.0)
    parser.add_argument("--x_max", type=float, default=1.2)
    parser.add_argument("--y_min", type=float, default=-0.6)
    parser.add_argument("--y_max", type=float, default=0.6)

    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    with open(args.ground_extrinsic, "r") as f:
        data = yaml.safe_load(f)

    H = load_matrix(data, "H_ground_to_undistorted_image")

    ground_pts = np.array([
        [args.x_max, args.y_min, 1.0],
        [args.x_max, args.y_max, 1.0],
        [args.x_min, args.y_max, 1.0],
        [args.x_min, args.y_min, 1.0],
    ], dtype=np.float64)

    img_pts_h = (H @ ground_pts.T).T
    img_pts = img_pts_h[:, :2] / img_pts_h[:, 2:3]
    img_pts = img_pts.astype(np.int32)

    vis = img.copy()

    cv2.polylines(vis, [img_pts], isClosed=True, color=(0, 0, 255), thickness=4)

    labels = ["front-left", "front-right", "back-right", "back-left"]
    for pt, label in zip(img_pts, labels):
        cv2.circle(vis, tuple(pt), 8, (0, 255, 0), -1)
        cv2.putText(vis, label, tuple(pt),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)

    print("Projected image points:")
    print(img_pts)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()