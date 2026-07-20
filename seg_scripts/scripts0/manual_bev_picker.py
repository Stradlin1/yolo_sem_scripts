#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Manual BEV point picker.

Usage example:
python3 manual_bev_picker.py \
  --image "$HOME/Desktop/relocate_ws/data/extracted/frames_undistorted/2026-06-28_01-05-54/frame_000000_xxx.jpg" \
  --out "$HOME/Desktop/relocate_ws/outputs/manual_bev.jpg" \
  --points_out "$HOME/Desktop/relocate_ws/outputs/manual_bev_points.yaml" \
  --h_out "$HOME/Desktop/relocate_ws/outputs/manual_bev_homography.yaml" \
  --out_w 320 --out_h 320

Mouse:
  Left click       : add a point, or select an existing point
  Drag point       : move selected point
  Right click      : delete nearest point

Keyboard:
  Enter            : generate BEV once
  r                : reset all points
  u                : undo last point
  q or Esc         : quit

Point order:
  1. BEV top-left
  2. BEV top-right
  3. BEV bottom-right
  4. BEV bottom-left

For ground images, this usually means:
  1-2 are the far edge of the ground patch,
  3-4 are the near edge of the ground patch.
"""

import os
import glob
import argparse
import yaml
import cv2
import numpy as np


class ManualBEVPicker:
    def __init__(self, image, display_scale):
        self.image = image
        self.h, self.w = image.shape[:2]
        self.display_scale = display_scale

        self.points = []  # original image coordinates, float32
        self.selected_idx = None
        self.dragging = False

        self.window_name = "manual_bev_picker"

    def image_to_display(self, pt):
        x, y = pt
        return int(round(x * self.display_scale)), int(round(y * self.display_scale))

    def display_to_image(self, x, y):
        return float(x / self.display_scale), float(y / self.display_scale)

    def nearest_point(self, x_img, y_img, threshold_px=15):
        if not self.points:
            return None

        threshold_img = threshold_px / self.display_scale
        pts = np.array(self.points, dtype=np.float32)
        d = np.sqrt((pts[:, 0] - x_img) ** 2 + (pts[:, 1] - y_img) ** 2)
        idx = int(np.argmin(d))

        if d[idx] <= threshold_img:
            return idx
        return None

    def mouse_callback(self, event, x, y, flags, param):
        x_img, y_img = self.display_to_image(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            idx = self.nearest_point(x_img, y_img)

            if idx is not None:
                self.selected_idx = idx
                self.dragging = True
            else:
                if len(self.points) < 4:
                    self.points.append([x_img, y_img])
                    self.selected_idx = len(self.points) - 1
                    self.dragging = True
                else:
                    print("[INFO] Already have 4 points. Drag existing points, or press r to reset.")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging and self.selected_idx is not None:
                x_img = min(max(x_img, 0.0), float(self.w - 1))
                y_img = min(max(y_img, 0.0), float(self.h - 1))
                self.points[self.selected_idx] = [x_img, y_img]

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.selected_idx = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            idx = self.nearest_point(x_img, y_img, threshold_px=20)
            if idx is not None:
                removed = self.points.pop(idx)
                print(f"[INFO] Removed point {idx + 1}: {removed}")

    def draw(self):
        if self.display_scale != 1.0:
            disp = cv2.resize(
                self.image,
                (int(round(self.w * self.display_scale)), int(round(self.h * self.display_scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            disp = self.image.copy()

        pts_disp = [self.image_to_display(p) for p in self.points]

        # Draw polygon lines
        if len(pts_disp) >= 2:
            for i in range(len(pts_disp) - 1):
                cv2.line(disp, pts_disp[i], pts_disp[i + 1], (0, 0, 255), 2)
            if len(pts_disp) == 4:
                cv2.line(disp, pts_disp[3], pts_disp[0], (0, 0, 255), 2)

        # Draw points and labels
        for i, pt in enumerate(pts_disp):
            cv2.circle(disp, pt, 7, (0, 255, 0), -1)
            cv2.circle(disp, pt, 11, (0, 0, 0), 2)
            cv2.putText(
                disp,
                str(i + 1),
                (pt[0] + 10, pt[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        help_lines = [
            "Click/drag 4 points: 1 TL, 2 TR, 3 BR, 4 BL",
            "Enter: generate BEV | r: reset | u: undo | right click: delete | q/Esc: quit",
            f"points: {len(self.points)}/4",
        ]

        y0 = 30
        for line in help_lines:
            cv2.putText(
                disp,
                line,
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                disp,
                line,
                (20, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y0 += 30

        return disp

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            disp = self.draw()
            cv2.imshow(self.window_name, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                cv2.destroyWindow(self.window_name)
                return None

            if key == ord("r"):
                self.points = []
                print("[INFO] Reset points.")

            if key == ord("u"):
                if self.points:
                    removed = self.points.pop()
                    print(f"[INFO] Undo point: {removed}")

            # Enter: 13 on many systems, 10 on some terminals/GTK backends
            if key in (13, 10):
                if len(self.points) != 4:
                    print("[WARN] Need exactly 4 points before generating BEV.")
                    continue

                cv2.destroyWindow(self.window_name)
                return np.array(self.points, dtype=np.float32)


def compute_display_scale(w, h, max_display_w, max_display_h):
    scale_w = max_display_w / float(w) if max_display_w > 0 else 1.0
    scale_h = max_display_h / float(h) if max_display_h > 0 else 1.0
    scale = min(1.0, scale_w, scale_h)
    return scale


def write_yaml_matrix(path, name, mat):
    data = {
        name: {
            "rows": int(mat.shape[0]),
            "cols": int(mat.shape[1]),
            "data": [float(x) for x in mat.reshape(-1)]
        }
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_points_yaml(path, src_pts, dst_pts, out_w, out_h):
    data = {
        "point_order": [
            "1: BEV top-left",
            "2: BEV top-right",
            "3: BEV bottom-right",
            "4: BEV bottom-left",
        ],
        "out_w": int(out_w),
        "out_h": int(out_h),
        "src_image_points": [[float(x), float(y)] for x, y in src_pts],
        "dst_bev_points": [[float(x), float(y)] for x, y in dst_pts],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def apply_homography_to_dir(input_dir, output_dir, H, out_w, out_h):
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(os.path.join(input_dir, "*.jpg")) +
        glob.glob(os.path.join(input_dir, "*.png")) +
        glob.glob(os.path.join(input_dir, "*.jpeg"))
    )

    if not image_paths:
        print(f"[WARN] No images found in batch input dir: {input_dir}")
        return

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] skip unreadable image: {path}")
            continue

        bev = cv2.warpPerspective(
            img,
            H,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        save_path = os.path.join(output_dir, os.path.basename(path))
        cv2.imwrite(save_path, bev)

        if i % 50 == 0:
            print(f"[{i}/{len(image_paths)}] {save_path}")

    print(f"[INFO] Batch BEV saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="One undistorted image used for manual point picking.")
    parser.add_argument("--out", required=True, help="Output BEV image path.")
    parser.add_argument("--out_w", type=int, default=320)
    parser.add_argument("--out_h", type=int, default=320)

    parser.add_argument("--points_out", default="", help="Optional YAML path to save selected points.")
    parser.add_argument("--h_out", default="", help="Optional YAML path to save image-to-BEV homography.")

    parser.add_argument("--max_display_w", type=int, default=1400)
    parser.add_argument("--max_display_h", type=int, default=900)

    parser.add_argument("--batch_input_dir", default="", help="Optional: apply selected H to all images in this dir.")
    parser.add_argument("--batch_output_dir", default="", help="Optional: output dir for batch BEV images.")

    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Cannot read image: {args.image}")

    h, w = img.shape[:2]
    display_scale = compute_display_scale(w, h, args.max_display_w, args.max_display_h)

    print(f"[INFO] Image: {args.image}")
    print(f"[INFO] Image size: {w} x {h}")
    print(f"[INFO] Display scale: {display_scale:.4f}")
    print("[INFO] Select 4 points in this order:")
    print("       1 = BEV top-left")
    print("       2 = BEV top-right")
    print("       3 = BEV bottom-right")
    print("       4 = BEV bottom-left")

    picker = ManualBEVPicker(img, display_scale)
    src_pts = picker.run()

    if src_pts is None:
        print("[INFO] Quit without generating BEV.")
        return

    dst_pts = np.array([
        [0, 0],
        [args.out_w - 1, 0],
        [args.out_w - 1, args.out_h - 1],
        [0, args.out_h - 1],
    ], dtype=np.float32)

    H_img_to_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)

    bev = cv2.warpPerspective(
        img,
        H_img_to_bev,
        (args.out_w, args.out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, bev)

    print("[INFO] Selected source points:")
    print(src_pts)
    print("[INFO] H_img_to_bev:")
    print(H_img_to_bev)
    print(f"[INFO] Saved BEV: {args.out}")

    if args.points_out:
        write_points_yaml(args.points_out, src_pts, dst_pts, args.out_w, args.out_h)
        print(f"[INFO] Saved points: {args.points_out}")

    if args.h_out:
        write_yaml_matrix(args.h_out, "H_img_to_bev", H_img_to_bev)
        print(f"[INFO] Saved homography: {args.h_out}")

    if args.batch_input_dir and args.batch_output_dir:
        apply_homography_to_dir(
            args.batch_input_dir,
            args.batch_output_dir,
            H_img_to_bev,
            args.out_w,
            args.out_h,
        )


if __name__ == "__main__":
    main()
