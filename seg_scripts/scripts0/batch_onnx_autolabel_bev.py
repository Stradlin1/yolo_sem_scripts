#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_CLASS_NAMES = [
    "background",
    "P",
    "A_DaTing",
    "B_TongDao",
    "C_ZhenLiaoShi",
    "JinQu",
]

# BGR
DEFAULT_COLORS = [
    (0, 0, 0),          # 0 background
    (230, 230, 230),    # 1 P
    (0, 220, 220),      # 2 A_DaTing
    (255, 180, 0),      # 3 B_TongDao
    (220, 0, 220),      # 4 C_ZhenLiaoShi
    (255, 0, 0),        # 5 JinQu
    (0, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (255, 255, 0),
]


def parse_class_names(text):
    names = [x.strip() for x in text.split(",") if x.strip()]
    if not names:
        raise ValueError("class names is empty")
    return names


def preprocess_bgr(img_bgr, input_size):
    img = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def softmax(x, axis=0):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def parse_output_to_mask(output, num_classes):
    """
    支持两类 ONNX 输出：

    A. 单通道类别 ID mask:
       [1, H, W]
       [H, W]

    B. 多通道 logits / probs:
       [1, C, H, W]
       [1, H, W, C]
       [C, H, W]
       [H, W, C]
    """
    out = np.asarray(output)

    # [1, H, W]
    if out.ndim == 3 and out.shape[0] == 1:
        mask_id = np.rint(out[0]).astype(np.uint8)
        return mask_id

    # [H, W]
    if out.ndim == 2:
        mask_id = np.rint(out).astype(np.uint8)
        return mask_id

    # [1, C, H, W] or [1, H, W, C]
    if out.ndim == 4:
        if out.shape[0] == 1 and out.shape[1] == num_classes:
            score = out[0]
        elif out.shape[0] == 1 and out.shape[-1] == num_classes:
            score = np.transpose(out[0], (2, 0, 1))
        else:
            raise RuntimeError(f"Unsupported 4D output shape: {out.shape}")
        prob = softmax(score, axis=0)
        return np.argmax(prob, axis=0).astype(np.uint8)

    # [C, H, W] or [H, W, C]
    if out.ndim == 3:
        if out.shape[0] == num_classes:
            score = out
        elif out.shape[-1] == num_classes:
            score = np.transpose(out, (2, 0, 1))
        else:
            raise RuntimeError(f"Unsupported 3D output shape: {out.shape}")
        prob = softmax(score, axis=0)
        return np.argmax(prob, axis=0).astype(np.uint8)

    raise RuntimeError(f"Unsupported output shape: {out.shape}")


def build_color_mask(mask_id, num_classes):
    h, w = mask_id.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(num_classes):
        color[mask_id == cls_id] = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]
    return color


def draw_class_text_on_regions(vis, mask_id, class_names, min_area=250):
    out = vis.copy()
    for cls_id, name in enumerate(class_names):
        if cls_id == 0:
            continue
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

            cv2.putText(out, text, (cx - 50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, text, (cx - 50, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_legend_panel(image, mask_id, class_names):
    h, w = image.shape[:2]
    panel_width = 460
    panel = np.full((h, panel_width, 3), 35, dtype=np.uint8)

    cv2.putText(panel, "ONNX BEV Segmentation", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"number of classes: {len(class_names)}", (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    unique_ids = sorted([int(x) for x in np.unique(mask_id)])
    unique_text = "pred ids: " + ",".join(map(str, unique_ids))
    cv2.putText(panel, unique_text[:42], (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

    total_pixels = mask_id.size
    y = 130
    for cls_id, name in enumerate(class_names):
        area = int(np.sum(mask_id == cls_id))
        ratio = area / total_pixels * 100.0
        color = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]

        cv2.rectangle(panel, (15, y - 16), (38, y + 7), color, -1)
        cv2.rectangle(panel, (15, y - 16), (38, y + 7), (255, 255, 255), 1)

        title = f"{cls_id}: {name}"
        info = f"pixels={area}  coverage={ratio:.2f}%"

        cv2.putText(panel, title, (52, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, info, (52, y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1, cv2.LINE_AA)

        y += 58

    return np.concatenate([image, panel], axis=1)


def make_debug_grid(original, color_mask, overlay_with_text):
    cv2.putText(original, "original", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(color_mask, "mask color", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay_with_text, "overlay + label", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return np.concatenate([original, color_mask, overlay_with_text], axis=1)


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
    mask_id:
      0 = background
      1 = P
      2 = A_DaTing
      3 = B_TongDao
      4 = C_ZhenLiaoShi
      5 = JinQu

    YOLO label:
      0 = P
      1 = A_DaTing
      2 = B_TongDao
      3 = C_ZhenLiaoShi
      4 = JinQu
    """
    h, w = mask_id.shape
    lines = []

    for mask_cls in range(1, 6):
        yolo_cls = mask_cls - 1
        bin_mask = (mask_id == mask_cls).astype(np.uint8)
        bin_mask = clean_class_mask(bin_mask, kernel_size=kernel_size, min_area=min_area)

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue

            line = contour_to_yolo_seg(
                contour=contour,
                cls_id=yolo_cls,
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


def iter_image_paths(folder):
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(sorted(folder.glob(ext)))
    return image_paths


def process_one_folder(
    sess,
    input_name,
    output_names,
    folder_name,
    bev_dir,
    out_dir,
    merged_images_dir,
    merged_labels_dir,
    class_names,
    input_size,
    alpha,
    min_area_text,
    min_area_label,
    kernel_size,
    epsilon_ratio,
):
    overlay_dir = out_dir / "overlay"
    mask_color_dir = out_dir / "mask_color"
    mask_id_dir = out_dir / "mask_id"
    grid_dir = out_dir / "grid"
    label_dir = out_dir / "labels"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_color_dir.mkdir(parents=True, exist_ok=True)
    mask_id_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = iter_image_paths(bev_dir)
    if not image_paths:
        print(f"[WARN] no images in {bev_dir}")
        return 0

    print(f"\nProcessing folder: {folder_name}")
    print(f"  BEV dir: {bev_dir}")
    print(f"  Images : {len(image_paths)}")

    count = 0
    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] failed to read {img_path}")
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        inp = preprocess_bgr(img_bgr, input_size)
        outputs = sess.run(output_names, {input_name: inp})
        mask_id = parse_output_to_mask(outputs[0], num_classes=len(class_names))

        mask_id = cv2.resize(mask_id, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        color_mask = build_color_mask(mask_id, len(class_names))
        overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, color_mask, alpha, 0)
        overlay_with_text = draw_class_text_on_regions(
            overlay, mask_id, class_names, min_area=min_area_text
        )
        overlay_with_legend = draw_legend_panel(overlay_with_text, mask_id, class_names)
        grid = make_debug_grid(img_bgr.copy(), color_mask.copy(), overlay_with_text.copy())

        stem = img_path.stem
        cv2.imwrite(str(mask_id_dir / f"{stem}.png"), mask_id)
        cv2.imwrite(str(mask_color_dir / f"{stem}.jpg"), color_mask)
        cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay_with_legend)
        cv2.imwrite(str(grid_dir / f"{stem}.jpg"), grid)

        label_path = label_dir / f"{stem}.txt"
        generate_yolo_labels(
            mask_id=mask_id,
            label_path=label_path,
            min_area=min_area_label,
            kernel_size=kernel_size,
            epsilon_ratio=epsilon_ratio,
        )

        # copy to merged dataset for CVAT/training
        merged_stem = f"{folder_name}__{stem}"
        merged_img_path = merged_images_dir / f"{merged_stem}{img_path.suffix.lower()}"
        merged_label_path = merged_labels_dir / f"{merged_stem}.txt"
        shutil.copy2(str(img_path), str(merged_img_path))
        shutil.copy2(str(label_path), str(merged_label_path))

        count += 1
        if idx == 0 or (idx + 1) % 20 == 0:
            unique_ids = sorted([int(x) for x in np.unique(mask_id)])
            print(f"  [{idx + 1}/{len(image_paths)}] {img_path.name}, pred ids={unique_ids}")

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model.")
    parser.add_argument("--bev-root", required=True, help="Root dir of bev_vehicle.")
    parser.add_argument("--output-root", required=True, help="Output root dir.")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--min-area-text", type=int, default=250)
    parser.add_argument("--min-area-label", type=int, default=80)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names.",
    )
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
    )
    parser.add_argument(
        "--folders",
        default="",
        help="Optional comma-separated subfolder names to process. "
             "If empty, process all subfolders under bev-root that contain a bev/ directory.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    bev_root = Path(args.bev_root)
    output_root = Path(args.output_root)

    class_names = parse_class_names(args.class_names)

    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(model_path), providers=providers)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    print("=" * 90)
    print("Batch ONNX BEV autolabel")
    print("Model      :", model_path)
    print("BEV root    :", bev_root)
    print("Output root :", output_root)
    print("Input name  :", input_name)
    print("Output names:", output_names)
    print("Providers   :", sess.get_providers())
    print("Class names :")
    for i, n in enumerate(class_names):
        print(f"  {i}: {n}")
    print("=" * 90)

    if args.folders.strip():
        folder_names = [x.strip() for x in args.folders.split(",") if x.strip()]
    else:
        folder_names = []
        for p in sorted(bev_root.iterdir()):
            if p.is_dir() and (p / "bev").exists():
                folder_names.append(p.name)

    if not folder_names:
        raise RuntimeError("No valid folders found.")

    merged_root = output_root / "merged_for_cvat"
    merged_images_dir = merged_root / "images"
    merged_labels_dir = merged_root / "labels"
    merged_images_dir.mkdir(parents=True, exist_ok=True)
    merged_labels_dir.mkdir(parents=True, exist_ok=True)

    # write classes.txt
    with open(merged_root / "classes.txt", "w", encoding="utf-8") as f:
        for name in class_names[1:]:
            f.write(name + "\n")

    # write a simple data.yaml
    with open(merged_root / "data.yaml", "w", encoding="utf-8") as f:
        f.write(f"path: {merged_root}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("test: images\n")
        f.write("\n")
        f.write("names:\n")
        for i, name in enumerate(class_names[1:]):
            f.write(f"  {i}: {name}\n")

    total = 0
    for folder_name in folder_names:
        bev_dir = bev_root / folder_name / "bev"
        if not bev_dir.exists():
            print(f"[WARN] skip {folder_name}, no bev dir")
            continue

        out_dir = output_root / folder_name
        count = process_one_folder(
            sess=sess,
            input_name=input_name,
            output_names=output_names,
            folder_name=folder_name,
            bev_dir=bev_dir,
            out_dir=out_dir,
            merged_images_dir=merged_images_dir,
            merged_labels_dir=merged_labels_dir,
            class_names=class_names,
            input_size=args.input_size,
            alpha=args.alpha,
            min_area_text=args.min_area_text,
            min_area_label=args.min_area_label,
            kernel_size=args.kernel_size,
            epsilon_ratio=args.epsilon_ratio,
        )
        total += count

    print("\n" + "=" * 90)
    print("Done.")
    print("Total images processed:", total)
    print("Per-folder outputs:")
    print("  <output-root>/<timestamp>/overlay")
    print("  <output-root>/<timestamp>/grid")
    print("  <output-root>/<timestamp>/labels")
    print("Merged dataset for CVAT:")
    print(f"  {merged_root}")
    print(f"  images: {merged_images_dir}")
    print(f"  labels: {merged_labels_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()