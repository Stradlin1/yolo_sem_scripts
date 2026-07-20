#!/usr/bin/env python3
import argparse
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
       每个像素已经是 class id。

    B. 多通道概率/logits:
       [1, C, H, W]
       [1, H, W, C]
       [C, H, W]
       [H, W, C]
       需要 argmax 得到 class id。
    """
    out = np.asarray(output)

    # 情况 A1: [1, H, W]，你的模型就是这种
    if out.ndim == 3 and out.shape[0] == 1:
        mask_id = out[0]

        # 防止输出是 float，但值本质上是类别 id
        mask_id = np.rint(mask_id).astype(np.uint8)
        return mask_id

    # 情况 A2: [H, W]
    if out.ndim == 2:
        mask_id = np.rint(out).astype(np.uint8)
        return mask_id

    # 情况 B: [1, C, H, W] 或 [1, H, W, C]
    if out.ndim == 4:
        if out.shape[0] == 1 and out.shape[1] == num_classes:
            score = out[0]  # [C,H,W]
        elif out.shape[0] == 1 and out.shape[-1] == num_classes:
            score = np.transpose(out[0], (2, 0, 1))  # [H,W,C] -> [C,H,W]
        else:
            raise RuntimeError(f"Unsupported 4D output shape: {out.shape}")

        prob = softmax(score, axis=0)
        mask_id = np.argmax(prob, axis=0).astype(np.uint8)
        return mask_id

    # 情况 B: [C, H, W] 或 [H, W, C]
    if out.ndim == 3:
        if out.shape[0] == num_classes:
            score = out
        elif out.shape[-1] == num_classes:
            score = np.transpose(out, (2, 0, 1))
        else:
            raise RuntimeError(f"Unsupported 3D output shape: {out.shape}")

        prob = softmax(score, axis=0)
        mask_id = np.argmax(prob, axis=0).astype(np.uint8)
        return mask_id

    raise RuntimeError(f"Unsupported output shape: {out.shape}")


def build_color_mask(mask_id, num_classes):
    h, w = mask_id.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for cls_id in range(num_classes):
        c = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]
        color[mask_id == cls_id] = c

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

            cv2.putText(
                out,
                text,
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                out,
                text,
                (cx - 50, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def draw_legend_panel(image, mask_id, class_names):
    h, w = image.shape[:2]
    panel_width = 460
    panel = np.full((h, panel_width, 3), 35, dtype=np.uint8)

    cv2.putText(
        panel,
        "ONNX BEV Segmentation",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"number of classes: {len(class_names)}",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    unique_ids = sorted([int(x) for x in np.unique(mask_id)])
    unique_text = "pred ids: " + ",".join(map(str, unique_ids))
    cv2.putText(
        panel,
        unique_text[:42],
        (15, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 220, 255),
        1,
        cv2.LINE_AA,
    )

    total_pixels = mask_id.size
    y = 130

    for cls_id, name in enumerate(class_names):
        area = int(np.sum(mask_id == cls_id))
        ratio = area / total_pixels * 100.0

        color = DEFAULT_COLORS[cls_id % len(DEFAULT_COLORS)]

        cv2.rectangle(panel, (15, y - 16), (38, y + 7), color, -1)
        cv2.rectangle(panel, (15, y - 16), (38, y + 7), (255, 255, 255), 1)

        title = f"{cls_id}: {name}"
        cv2.putText(
            panel,
            title,
            (52, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        info = f"pixels={area}  coverage={ratio:.2f}%"
        cv2.putText(
            panel,
            info,
            (52, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        y += 58

    return np.concatenate([image, panel], axis=1)


def make_debug_grid(original, color_mask, overlay_with_text):
    h, w = original.shape[:2]

    cv2.putText(
        original,
        "original",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        color_mask,
        "mask color",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        overlay_with_text,
        "overlay + label",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.concatenate([original, color_mask, overlay_with_text], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--min-area-text", type=int, default=250)
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASS_NAMES),
        help="Comma-separated class names. Example: background,P,A_DaTing,B_TongDao,C_ZhenLiaoShi,JinQu",
    )
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    overlay_dir = output_dir / "overlay"
    mask_color_dir = output_dir / "mask_color"
    mask_id_dir = output_dir / "mask_id"
    grid_dir = output_dir / "grid"

    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_color_dir.mkdir(parents=True, exist_ok=True)
    mask_id_dir.mkdir(parents=True, exist_ok=True)
    grid_dir.mkdir(parents=True, exist_ok=True)

    class_names = parse_class_names(args.class_names)
    num_classes = len(class_names)

    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]

    if not providers:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(model_path), providers=providers)

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    print("=" * 80)
    print("ONNX BEV segmentation visualization")
    print("Model:", model_path)
    print("Input:", input_dir)
    print("Output:", output_dir)
    print("Input name:", input_name)
    print("Output names:", output_names)
    print("Providers:", sess.get_providers())
    print("Class names:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print("=" * 80)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(sorted(input_dir.glob(ext)))

    if not image_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    print("Images:", len(image_paths))

    first_shape_printed = False

    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img_bgr is None:
            print("[WARN] failed to read:", img_path)
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        inp = preprocess_bgr(img_bgr, args.input_size)
        outputs = sess.run(output_names, {input_name: inp})

        if not first_shape_printed:
            print("Raw output shapes:")
            for name, out in zip(output_names, outputs):
                arr = np.asarray(out)
                print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}")
            first_shape_printed = True

        mask_id = parse_output_to_mask(outputs[0], num_classes=num_classes)

        mask_id = cv2.resize(
            mask_id,
            (orig_w, orig_h),
            interpolation=cv2.INTER_NEAREST,
        )

        unique_ids = sorted([int(x) for x in np.unique(mask_id)])
        invalid_ids = [x for x in unique_ids if x < 0 or x >= num_classes]
        if invalid_ids:
            print(f"[WARN] {img_path.name} has invalid class ids: {invalid_ids}")

        color_mask = build_color_mask(mask_id, num_classes)

        overlay = cv2.addWeighted(
            img_bgr,
            1.0 - args.alpha,
            color_mask,
            args.alpha,
            0,
        )

        overlay_with_text = draw_class_text_on_regions(
            overlay,
            mask_id,
            class_names,
            min_area=args.min_area_text,
        )

        overlay_with_legend = draw_legend_panel(
            overlay_with_text,
            mask_id,
            class_names,
        )

        grid = make_debug_grid(
            img_bgr.copy(),
            color_mask.copy(),
            overlay_with_text.copy(),
        )

        stem = img_path.stem

        cv2.imwrite(str(mask_id_dir / f"{stem}.png"), mask_id)
        cv2.imwrite(str(mask_color_dir / f"{stem}.jpg"), color_mask)
        cv2.imwrite(str(overlay_dir / f"{stem}.jpg"), overlay_with_legend)
        cv2.imwrite(str(grid_dir / f"{stem}.jpg"), grid)

        if idx == 0 or (idx + 1) % 20 == 0:
            print(f"[{idx + 1}/{len(image_paths)}] {img_path.name}, pred ids={unique_ids}")

    print("=" * 80)
    print("Done.")
    print("Overlay with legend:", overlay_dir)
    print("Color masks:", mask_color_dir)
    print("ID masks:", mask_id_dir)
    print("Debug grids:", grid_dir)
    print("=" * 80)


if __name__ == "__main__":
    main()