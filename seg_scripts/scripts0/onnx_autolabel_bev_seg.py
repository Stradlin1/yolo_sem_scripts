#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = [
    "P",
    "A_DaTing",
    "B_TongDao",
    "C_ZhenLiaoShi",
    "JinQu",
]

# 可视化颜色，BGR
COLORS = {
    0: (0, 0, 0),          # background
    1: (220, 220, 220),    # P
    2: (255, 200, 0),      # A_DaTing
    3: (0, 220, 220),      # B_TongDao
    4: (220, 0, 220),      # C_ZhenLiaoShi
    5: (255, 0, 0),        # JinQu
}


def preprocess_bgr(img_bgr, input_size):
    img = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def parse_output(output):
    """
    支持常见语义分割输出：
    1. [1, C, H, W]
    2. [1, H, W, C]
    3. [C, H, W]
    4. [H, W, C]

    返回:
        prob: [C, H, W]
    """
    out = output

    if isinstance(out, list):
        out = out[0]

    out = np.asarray(out)

    if out.ndim == 4:
        # [1, C, H, W]
        if out.shape[0] == 1 and out.shape[1] <= 20:
            out = out[0]
        # [1, H, W, C]
        elif out.shape[0] == 1 and out.shape[-1] <= 20:
            out = np.transpose(out[0], (2, 0, 1))
        else:
            raise RuntimeError(f"Unsupported output shape: {out.shape}")

    elif out.ndim == 3:
        # [C, H, W]
        if out.shape[0] <= 20:
            pass
        # [H, W, C]
        elif out.shape[-1] <= 20:
            out = np.transpose(out, (2, 0, 1))
        else:
            raise RuntimeError(f"Unsupported output shape: {out.shape}")
    else:
        raise RuntimeError(f"Unsupported output shape: {out.shape}")

    # 如果像 logits，做 softmax；如果已经像概率，也不影响太大
    prob = softmax(out, axis=0)
    return prob


def mask_to_color(mask_id):
    h, w = mask_id.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for k, color in COLORS.items():
        vis[mask_id == k] = color
    return vis


def clean_class_mask(bin_mask, kernel_size=3, min_area=50):
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


def generate_yolo_labels(
    mask_id,
    label_path,
    min_area=80,
    kernel_size=3,
    epsilon_ratio=0.002,
):
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
                contour,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model.")
    parser.add_argument("--input", required=True, help="Input BEV image directory.")
    parser.add_argument("--output", required=True, help="Output autolabel directory.")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--conf-thres", type=float, default=0.45)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--epsilon-ratio", type=float, default=0.002)
    parser.add_argument("--providers", default="CUDAExecutionProvider,CPUExecutionProvider")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    mask_id_dir = output_dir / "masks_id"
    mask_color_dir = output_dir / "masks_color"
    label_dir = output_dir / "labels"

    mask_id_dir.mkdir(parents=True, exist_ok=True)
    mask_color_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(args.model, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    print("ONNX model:", args.model)
    print("Input:", input_dir)
    print("Output:", output_dir)
    print("Input name:", input_name)
    print("Output names:", output_names)
    print("Providers:", sess.get_providers())

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(sorted(input_dir.glob(ext)))

    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    print("Images:", len(image_paths))

    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[WARN] Failed to read {img_path}")
            continue

        inp = preprocess_bgr(img_bgr, args.input_size)
        outputs = sess.run(output_names, {input_name: inp})

        prob = parse_output(outputs[0])
        pred_score = np.max(prob, axis=0)
        pred_cls = np.argmax(prob, axis=0).astype(np.uint8)

        # 这里假设模型输出包含 background + 5 类，即 6 通道：
        # 0 background, 1 P, 2 A_DaTing, 3 B_TongDao, 4 C_ZhenLiaoShi, 5 JinQu
        # 如果你的模型没有 background，需要按实际情况改这里。
        mask_id = pred_cls.copy()
        mask_id[pred_score < args.conf_thres] = 0

        mask_id = cv2.resize(
            mask_id,
            (img_bgr.shape[1], img_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        stem = img_path.stem

        cv2.imwrite(str(mask_id_dir / f"{stem}.png"), mask_id)

        color = mask_to_color(mask_id)
        overlay = cv2.addWeighted(img_bgr, 0.55, color, 0.45, 0)
        cv2.imwrite(str(mask_color_dir / f"{stem}.jpg"), overlay)

        generate_yolo_labels(
            mask_id=mask_id,
            label_path=label_dir / f"{stem}.txt",
            min_area=args.min_area,
            kernel_size=args.kernel_size,
            epsilon_ratio=args.epsilon_ratio,
        )

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"[{idx + 1}/{len(image_paths)}] {img_path.name}")

    print("Done.")
    print("Mask id dir:", mask_id_dir)
    print("Mask color dir:", mask_color_dir)
    print("YOLO label dir:", label_dir)


if __name__ == "__main__":
    main()