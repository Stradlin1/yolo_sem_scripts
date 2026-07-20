#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


# mask id == YOLO id
# background 用 255，不进入 YOLO 标签
CLASS_NAMES = [
    "P",
    "A_DaTing",
    "B_TongDao",
    "C_ZhenLiaoShi",
    "JinQu",
]

BACKGROUND_ID = 255

# OpenCV uses BGR.
# RGB names:
# 0 P              RGB=(255,255,255) white
# 1 A_DaTing       RGB=(0,255,255)   cyan
# 2 B_TongDao      RGB=(255,165,0)   orange
# 3 C_ZhenLiaoShi  RGB=(255,0,255)   magenta
# 4 JinQu          RGB=(0,0,255)     blue
# 255 background   RGB=(35,35,35)    dark gray
COLORS_BGR = {
    0: (255, 255, 255),   # P, white
    1: (255, 255, 0),     # A_DaTing, cyan
    2: (0, 165, 255),     # B_TongDao, orange
    3: (255, 0, 255),     # C_ZhenLiaoShi, magenta
    4: (255, 0, 0),       # JinQu, blue
    BACKGROUND_ID: (35, 35, 35),  # background, dark gray
}

COLOR_NAMES = {
    0: "white",
    1: "cyan",
    2: "orange",
    3: "magenta",
    4: "blue",
    BACKGROUND_ID: "dark gray",
}

RGB_VALUES = {
    0: (255, 255, 255),
    1: (0, 255, 255),
    2: (255, 165, 0),
    3: (255, 0, 255),
    4: (0, 0, 255),
    BACKGROUND_ID: (35, 35, 35),
}


def list_images(image_dir):
    paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        paths.extend(sorted(Path(image_dir).glob(ext)))
    return paths


def remap_old_onnx_mask_to_new(mask):
    """
    旧 ONNX mask:
      0 background
      1 P
      2 A_DaTing
      3 B_TongDao
      4 C_ZhenLiaoShi
      5 JinQu

    新 mask:
      0 P
      1 A_DaTing
      2 B_TongDao
      3 C_ZhenLiaoShi
      4 JinQu
      255 background
    """
    unique_ids = set(int(x) for x in np.unique(mask))

    # 已经是新格式：包含 255，或最大值 <= 4 且没有旧格式 5
    if BACKGROUND_ID in unique_ids:
        return mask.astype(np.uint8)

    # 默认把从 ONNX autolabel 读进来的 0..5 旧 mask 转成新格式
    if unique_ids.issubset({0, 1, 2, 3, 4, 5}):
        new_mask = np.full(mask.shape, BACKGROUND_ID, dtype=np.uint8)
        new_mask[mask == 1] = 0
        new_mask[mask == 2] = 1
        new_mask[mask == 3] = 2
        new_mask[mask == 4] = 3
        new_mask[mask == 5] = 4
        return new_mask

    return mask.astype(np.uint8)


def colorize_mask(mask_id):
    h, w = mask_id.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    color[:, :] = COLORS_BGR[BACKGROUND_ID]

    for cls_id in range(len(CLASS_NAMES)):
        color[mask_id == cls_id] = COLORS_BGR[cls_id]

    return color


def make_overlay(img, mask_id, alpha=0.45):
    color = colorize_mask(mask_id)
    return cv2.addWeighted(img, 1.0 - alpha, color, alpha, 0)


def draw_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_region_labels(vis, mask_id, min_area=250):
    out = vis.copy()

    for cls_id, name in enumerate(CLASS_NAMES):
        bin_mask = (mask_id == cls_id).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            cx, cy = centroids[i]
            cx = int(cx)
            cy = int(cy)

            text = f"{cls_id}:{name}"

            cv2.putText(
                out,
                text,
                (cx - 55, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (cx - 55, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def draw_color_legend(mask_id, height, width=500):
    """
    右侧 legend 压缩行高，保证 320 高度也能完整显示 5 类 + background。
    """
    panel = np.full((height, width, 3), 35, dtype=np.uint8)
    total_pixels = mask_id.size

    cv2.putText(
        panel,
        "class color legend",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        "mask id == YOLO id, bg=255",
        (12, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    unique_ids = sorted([int(x) for x in np.unique(mask_id)])
    pred_text = "pred ids: " + ",".join(map(str, unique_ids))
    cv2.putText(
        panel,
        pred_text[:55],
        (12, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    rows = list(range(len(CLASS_NAMES))) + [BACKGROUND_ID]

    y = 98
    row_h = 36

    for cls_id in rows:
        if cls_id == BACKGROUND_ID:
            name = "background"
        else:
            name = CLASS_NAMES[cls_id]

        color_bgr = COLORS_BGR[cls_id]
        color_name = COLOR_NAMES[cls_id]
        rgb = RGB_VALUES[cls_id]

        area = int(np.sum(mask_id == cls_id))
        ratio = area / total_pixels * 100.0

        cv2.rectangle(panel, (12, y - 14), (38, y + 10), color_bgr, -1)
        cv2.rectangle(panel, (12, y - 14), (38, y + 10), (255, 255, 255), 1)

        if cls_id == BACKGROUND_ID:
            line1 = f"255: background  {color_name}"
        else:
            line1 = f"{cls_id}: {name}  {color_name}"

        line2 = f"RGB={rgb}  {ratio:.1f}%"

        cv2.putText(
            panel,
            line1,
            (48, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            line2,
            (48, y + 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        y += row_h

    return panel


def draw_hud(width, current_class, brush_size, idx, total, image_name, dirty):
    panel_h = 105
    panel = np.full((panel_h, width, 3), 35, dtype=np.uint8)

    status = "UNSAVED" if dirty else "saved"
    status_color = (0, 180, 255) if dirty else (180, 255, 180)

    line1 = f"[{idx + 1}/{total}] {image_name}    {status}"
    line2 = f"current: {current_class}:{CLASS_NAMES[current_class]}    brush: {brush_size}"
    line3 = "0 P | 1 A_DaTing | 2 B_TongDao | 3 C_ZhenLiaoShi | 4 JinQu | right mouse = background"
    line4 = "left draw | right erase | s save | n next | p prev | [ ] brush | c clear bg | r reload | q quit"

    cv2.putText(panel, line1, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 1, cv2.LINE_AA)
    cv2.putText(panel, line2, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLORS_BGR[current_class], 2, cv2.LINE_AA)
    cv2.putText(panel, line3, (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, line4, (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)

    return panel


def compose_editor_view(img, mask_id, alpha, current_class, brush_size, idx, total, image_name, dirty, min_area_text):
    overlay_edit = make_overlay(img, mask_id, alpha=alpha)
    overlay_label = draw_region_labels(overlay_edit, mask_id, min_area=min_area_text)

    overlay_edit = draw_title(overlay_edit, "edit overlay - draw here")
    overlay_label = draw_title(overlay_label, "overlay + label names")
    legend = draw_color_legend(mask_id, height=img.shape[0], width=500)

    body = np.hstack([overlay_edit, overlay_label, legend])

    hud = draw_hud(
        width=body.shape[1],
        current_class=current_class,
        brush_size=brush_size,
        idx=idx,
        total=total,
        image_name=image_name,
        dirty=dirty,
    )

    return np.vstack([hud, body])


def clean_class_mask(bin_mask, kernel_size=3, min_area=80):
    bin_mask = bin_mask.astype(np.uint8)

    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
    clean = np.zeros_like(bin_mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 1

    return clean


def contour_to_yolo_seg(contour, cls_id, width, height, epsilon_ratio=0.002):
    area = cv2.contourArea(contour)
    if area <= 1:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        return None

    points = approx.reshape(-1, 2)
    values = [str(cls_id)]

    for x, y in points:
        x_norm = np.clip(float(x) / float(width), 0.0, 1.0)
        y_norm = np.clip(float(y) / float(height), 0.0, 1.0)
        values.append(f"{x_norm:.6f}")
        values.append(f"{y_norm:.6f}")

    return " ".join(values)


def generate_yolo_labels(mask_id, label_path, min_area=80, kernel_size=3, epsilon_ratio=0.002):
    """
    新格式:
      mask id == YOLO id
      0 P
      1 A_DaTing
      2 B_TongDao
      3 C_ZhenLiaoShi
      4 JinQu
      255 background, ignored
    """
    h, w = mask_id.shape
    lines = []

    for cls_id in range(len(CLASS_NAMES)):
        bin_mask = (mask_id == cls_id).astype(np.uint8)
        bin_mask = clean_class_mask(
            bin_mask,
            kernel_size=kernel_size,
            min_area=min_area,
        )

        contours, _ = cv2.findContours(
            bin_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            line = contour_to_yolo_seg(
                contour=contour,
                cls_id=cls_id,
                width=w,
                height=h,
                epsilon_ratio=epsilon_ratio,
            )

            if line is not None:
                lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


class MaskEditor:
    def __init__(self, args):
        self.image_dir = Path(args.images)
        self.mask_dir = Path(args.masks)
        self.out_mask_dir = Path(args.out_masks)
        self.out_label_dir = Path(args.out_labels)
        self.out_overlay_dir = Path(args.out_overlays)

        self.alpha = args.alpha
        self.brush_size = args.brush_size
        self.current_class = args.start_class
        self.min_area = args.min_area
        self.kernel_size = args.kernel_size
        self.epsilon_ratio = args.epsilon_ratio
        self.min_area_text = args.min_area_text

        self.images = list_images(self.image_dir)
        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.out_mask_dir.mkdir(parents=True, exist_ok=True)
        self.out_label_dir.mkdir(parents=True, exist_ok=True)
        self.out_overlay_dir.mkdir(parents=True, exist_ok=True)

        self.idx = 0
        self.img = None
        self.mask = None
        self.dirty = False
        self.drawing = False
        self.erase_mode = False

        self.window_name = "OpenCV BEV Label Editor"

    def load_current(self):
        img_path = self.images[self.idx]
        stem = img_path.stem

        self.img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        edited_mask_path = self.out_mask_dir / f"{stem}.png"
        raw_mask_path = self.mask_dir / f"{stem}.png"

        # 已修过的优先读取；未修过的读取 ONNX 旧格式 mask 并重映射
        if edited_mask_path.exists():
            mask_path = edited_mask_path
            is_edited = True
        else:
            mask_path = raw_mask_path
            is_edited = False

        if not mask_path.exists():
            raise RuntimeError(f"Mask not found for {img_path.name}: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = mask.astype(np.uint8)

        if not is_edited:
            mask = remap_old_onnx_mask_to_new(mask)

        h, w = self.img.shape[:2]
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        self.mask = mask
        self.dirty = False

    def save_current(self):
        img_path = self.images[self.idx]
        stem = img_path.stem

        mask_path = self.out_mask_dir / f"{stem}.png"
        label_path = self.out_label_dir / f"{stem}.txt"
        overlay_path = self.out_overlay_dir / f"{stem}.jpg"

        cv2.imwrite(str(mask_path), self.mask)

        generate_yolo_labels(
            mask_id=self.mask,
            label_path=label_path,
            min_area=self.min_area,
            kernel_size=self.kernel_size,
            epsilon_ratio=self.epsilon_ratio,
        )

        overlay = make_overlay(self.img, self.mask, alpha=self.alpha)
        overlay_label = draw_region_labels(overlay, self.mask, min_area=self.min_area_text)
        cv2.imwrite(str(overlay_path), overlay_label)

        self.dirty = False

        print(f"[SAVE] {img_path.name}")
        print(f"       mask : {mask_path}")
        print(f"       label: {label_path}")
        print(f"       overlay: {overlay_path}")

    def render(self):
        canvas = compose_editor_view(
            img=self.img,
            mask_id=self.mask,
            alpha=self.alpha,
            current_class=self.current_class,
            brush_size=self.brush_size,
            idx=self.idx,
            total=len(self.images),
            image_name=self.images[self.idx].name,
            dirty=self.dirty,
            min_area_text=self.min_area_text,
        )
        cv2.imshow(self.window_name, canvas)

    def paint(self, x, y, cls_id):
        h, w = self.mask.shape[:2]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        cv2.circle(
            self.mask,
            (x, y),
            self.brush_size,
            int(cls_id),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        self.dirty = True

    def mouse_callback(self, event, x, y, flags, param):
        hud_h = 105
        img_h, img_w = self.img.shape[:2]

        y_img = y - hud_h

        if y_img < 0 or y_img >= img_h:
            return

        # 只允许在左侧编辑图绘制
        if x < 0 or x >= img_w:
            return

        x_img = x

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.erase_mode = False
            self.paint(x_img, y_img, self.current_class)

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.drawing = True
            self.erase_mode = True
            self.paint(x_img, y_img, BACKGROUND_ID)

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                if self.erase_mode:
                    self.paint(x_img, y_img, BACKGROUND_ID)
                else:
                    self.paint(x_img, y_img, self.current_class)

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            self.drawing = False
            self.erase_mode = False

    def next_image(self):
        if self.dirty:
            print("[WARN] unsaved changes. Press s to save before moving to next.")
            return

        self.idx = min(self.idx + 1, len(self.images) - 1)
        self.load_current()

    def prev_image(self):
        if self.dirty:
            print("[WARN] unsaved changes. Press s to save before moving to previous.")
            return

        self.idx = max(self.idx - 1, 0)
        self.load_current()

    def run(self):
        self.load_current()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1500, 650)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("=" * 80)
        print("OpenCV BEV Label Editor")
        print("")
        print("ID mapping:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  mask {i} == yolo {i}: {name}")
        print(f"  mask {BACKGROUND_ID}: background, ignored by YOLO")
        print("")
        print("RGB colors:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {i}: {name:14s} RGB={RGB_VALUES[i]} {COLOR_NAMES[i]}")
        print(f"  255: {'background':14s} RGB={RGB_VALUES[BACKGROUND_ID]} {COLOR_NAMES[BACKGROUND_ID]}")
        print("")
        print("Controls:")
        print("  0: P")
        print("  1: A_DaTing")
        print("  2: B_TongDao")
        print("  3: C_ZhenLiaoShi")
        print("  4: JinQu")
        print("  left mouse : draw current class on left panel")
        print("  right mouse: erase to background=255 on left panel")
        print("  [: smaller brush")
        print("  ]: larger brush")
        print("  s: save mask + yolo label")
        print("  n: next image")
        print("  p: previous image")
        print("  c: clear all to background")
        print("  r: reload current image")
        print("  q or ESC: quit")
        print("=" * 80)

        while True:
            self.render()
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key == ord("q") or key == 27:
                if self.dirty:
                    print("[WARN] unsaved changes. Press s to save before quitting.")
                    continue
                break

            elif key in [ord("0"), ord("1"), ord("2"), ord("3"), ord("4")]:
                self.current_class = int(chr(key))
                print(f"[CLASS] {self.current_class}: {CLASS_NAMES[self.current_class]}")

            elif key == ord("["):
                self.brush_size = max(1, self.brush_size - 2)
                print(f"[BRUSH] {self.brush_size}")

            elif key == ord("]"):
                self.brush_size = min(100, self.brush_size + 2)
                print(f"[BRUSH] {self.brush_size}")

            elif key == ord("s"):
                self.save_current()

            elif key == ord("n"):
                self.next_image()

            elif key == ord("p"):
                self.prev_image()

            elif key == ord("c"):
                self.mask[:, :] = BACKGROUND_ID
                self.dirty = True
                print("[CLEAR] mask set to background=255")

            elif key == ord("r"):
                self.load_current()
                print("[RELOAD] current image")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="BEV image directory.")
    parser.add_argument("--masks", required=True, help="Input mask_id directory from ONNX.")
    parser.add_argument("--out-masks", required=True, help="Output edited mask_id directory.")
    parser.add_argument("--out-labels", required=True, help="Output edited YOLO-seg labels directory.")
    parser.add_argument("--out-overlays", required=True, help="Output edited overlay directory.")
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--brush-size", type=int, default=8)
    parser.add_argument("--start-class", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)
    parser.add_argument("--min-area-text", type=int, default=250)
    args = parser.parse_args()

    if args.start_class < 0 or args.start_class >= len(CLASS_NAMES):
        raise ValueError("--start-class must be 0..4")

    editor = MaskEditor(args)
    editor.run()


if __name__ == "__main__":
    main()