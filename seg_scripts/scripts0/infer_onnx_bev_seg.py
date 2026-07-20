#!/usr/bin/env python3

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

TRAVERSABLE_IDS = [2, 3, 4]  # B_TongDao, A_DaTing, P


def collect_images(input_path):
    p = Path(input_path)

    if p.is_file():
        return [str(p)]

    if p.is_dir():
        paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            paths.extend(glob.glob(str(p / ext)))
            paths.extend(glob.glob(str(p / ext.upper())))
        return sorted(paths)

    return sorted(glob.glob(str(input_path)))


def preprocess_bgr(img_bgr, input_size=320):
    """
    BEV image -> ONNX input.
    Assumption:
      - model input is RGB
      - normalized to 0~1
      - shape is NCHW: [1, 3, 320, 320]
    """
    if img_bgr.shape[0] != input_size or img_bgr.shape[1] != input_size:
        img_bgr = cv2.resize(
            img_bgr,
            (input_size, input_size),
            interpolation=cv2.INTER_LINEAR,
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # CHW -> NCHW
    return x


def decode_semantic_output(output, num_classes=6, verbose=False):
    """
    Convert ONNX output to 2D class map [H, W].

    Supported:
      [1, C, H, W]  -> argmax over C
      [1, H, W, C]  -> argmax over C
      [C, H, W]     -> argmax over C
      [H, W, C]     -> argmax over C
      [1, H, W]     -> already class map
      [H, W]        -> already class map
    """
    arr = np.asarray(output)

    if verbose:
        print(f"[DEBUG] raw output shape: {arr.shape}, dtype: {arr.dtype}")

    if arr.ndim == 4:
        # NCHW: [1, C, H, W]
        if arr.shape[0] == 1 and arr.shape[1] == num_classes:
            cls_map = np.argmax(arr, axis=1)[0]

        # NHWC: [1, H, W, C]
        elif arr.shape[0] == 1 and arr.shape[-1] == num_classes:
            cls_map = np.argmax(arr, axis=-1)[0]

        else:
            raise RuntimeError(
                f"Unsupported 4D output shape: {arr.shape}. "
                f"Expected [1,{num_classes},H,W] or [1,H,W,{num_classes}]."
            )

    elif arr.ndim == 3:
        # [1, H, W], already class map
        if arr.shape[0] == 1:
            cls_map = arr[0]

        # [C, H, W], logits without batch
        elif arr.shape[0] == num_classes:
            cls_map = np.argmax(arr, axis=0)

        # [H, W, C], logits without batch
        elif arr.shape[-1] == num_classes:
            cls_map = np.argmax(arr, axis=-1)

        else:
            raise RuntimeError(
                f"Unsupported 3D output shape: {arr.shape}. "
                f"Expected [1,H,W], [{num_classes},H,W], or [H,W,{num_classes}]."
            )

    elif arr.ndim == 2:
        # [H, W], already class map
        cls_map = arr

    else:
        raise RuntimeError(f"Unsupported output shape: {arr.shape}")

    cls_map = cls_map.astype(np.uint8)

    if verbose:
        unique_ids, counts = np.unique(cls_map, return_counts=True)
        stat = ", ".join(f"{int(i)}={int(c)}" for i, c in zip(unique_ids, counts))
        print(f"[DEBUG] class map shape: {cls_map.shape}, unique/count: {stat}")

    return cls_map


def colorize_class_map(cls_map):
    """
    Return BGR visualization.
    """
    colors = {
        0: (0, 0, 255),       # JinQu
        1: (255, 0, 0),       # C_ZhenLiaoShi
        2: (0, 255, 0),       # B_TongDao
        3: (0, 255, 255),     # A_DaTing
        4: (255, 255, 255),   # P
        5: (0, 0, 0),         # background
    }

    if cls_map.ndim != 2:
        raise RuntimeError(f"cls_map must be 2D, got shape {cls_map.shape}")

    vis = np.zeros((cls_map.shape[0], cls_map.shape[1], 3), dtype=np.uint8)

    for cls_id, color in colors.items():
        vis[cls_map == cls_id] = color

    return vis


def make_overlay(img_bgr, sem_vis, alpha=0.45):
    img_bgr = cv2.resize(
        img_bgr,
        (sem_vis.shape[1], sem_vis.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.addWeighted(img_bgr, 1.0 - alpha, sem_vis, alpha, 0)


def save_legend(out_dir):
    legend_path = os.path.join(out_dir, "class_legend.txt")
    with open(legend_path, "w", encoding="utf-8") as f:
        for cls_id, name in CLASS_NAMES.items():
            f.write(f"{cls_id}: {name}\n")
        f.write("\nTraversable ids: 2, 3, 4 = B_TongDao, A_DaTing, P\n")


def ensure_output_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "class_map"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "semantic_vis"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "traversable_mask"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "overlay"), exist_ok=True)


def print_class_stat(cls_map):
    unique_ids, counts = np.unique(cls_map, return_counts=True)
    parts = []
    for cls_id, count in zip(unique_ids, counts):
        cls_id = int(cls_id)
        count = int(count)
        name = CLASS_NAMES.get(cls_id, "unknown")
        parts.append(f"{cls_id}:{name}={count}")
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to exp.onnx")
    parser.add_argument("--input", required=True, help="BEV image file, folder, or glob")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--input-size", type=int, default=320)
    parser.add_argument("--debug", action="store_true", help="Print output tensor shape for every image")
    args = parser.parse_args()

    ensure_output_dirs(args.output)
    save_legend(args.output)

    image_paths = collect_images(args.input)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found: {args.input}")

    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers)

    input_meta = sess.get_inputs()[0]
    output_metas = sess.get_outputs()

    input_name = input_meta.name
    output_names = [o.name for o in output_metas]

    print("========================================")
    print("ONNX BEV semantic inference")
    print(f"Model:        {args.model}")
    print(f"Input:        {args.input}")
    print(f"Output:       {args.output}")
    print(f"Input name:   {input_name}")
    print(f"Input shape:  {input_meta.shape}")
    print(f"Output names: {output_names}")
    for o in output_metas:
        print(f"Output meta:  name={o.name}, shape={o.shape}, type={o.type}")
    print(f"Images:       {len(image_paths)}")
    print("========================================")

    ok_count = 0

    for idx, img_path in enumerate(image_paths):
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img_bgr is None:
            print(f"[WARN] failed to read: {img_path}")
            continue

        x = preprocess_bgr(img_bgr, args.input_size)

        # Run all outputs. Usually there is only output0.
        outputs = sess.run(None, {input_name: x})

        # Try to decode the first output. If it fails, try the rest.
        last_error = None
        cls_map = None

        for out_i, out in enumerate(outputs):
            try:
                verbose = args.debug or (idx == 0)
                cls_map = decode_semantic_output(
                    out,
                    num_classes=len(CLASS_NAMES),
                    verbose=verbose,
                )
                if idx == 0:
                    print(f"[INFO] Using ONNX output index {out_i}: {output_names[out_i]}")
                break
            except Exception as e:
                last_error = e

        if cls_map is None:
            raise RuntimeError(f"Failed to decode ONNX outputs. Last error: {last_error}")

        # Make sure result is input-size image.
        if cls_map.shape != (args.input_size, args.input_size):
            cls_map = cv2.resize(
                cls_map,
                (args.input_size, args.input_size),
                interpolation=cv2.INTER_NEAREST,
            )

        sem_vis = colorize_class_map(cls_map)

        traversable_mask = np.isin(cls_map, TRAVERSABLE_IDS).astype(np.uint8) * 255

        overlay = make_overlay(img_bgr, sem_vis)

        stem = Path(img_path).stem

        class_map_path = os.path.join(args.output, "class_map", f"{stem}_class.png")
        sem_vis_path = os.path.join(args.output, "semantic_vis", f"{stem}_sem.png")
        mask_path = os.path.join(args.output, "traversable_mask", f"{stem}_trav.png")
        overlay_path = os.path.join(args.output, "overlay", f"{stem}_overlay.png")

        cv2.imwrite(class_map_path, cls_map)
        cv2.imwrite(sem_vis_path, sem_vis)
        cv2.imwrite(mask_path, traversable_mask)
        cv2.imwrite(overlay_path, overlay)

        print(f"[OK] {img_path}")
        print(f"     classes: {print_class_stat(cls_map)}")

        ok_count += 1

    print("========================================")
    print(f"[DONE] {ok_count}/{len(image_paths)} images processed.")
    print(f"Results saved to: {args.output}")
    print("========================================")


if __name__ == "__main__":
    main()
