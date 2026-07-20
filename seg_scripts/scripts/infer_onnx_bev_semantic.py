#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run semantic-segmentation ONNX inference on BEV images.

Expected model:
  input : float32 [1, 3, 320, 320]
  output: uint8   [1, 320, 320] or [320, 320]

Output:
  masks_id/   single-channel PNG masks, pixel IDs:
                0 background
                1 P
                2 A_DaTing
                3 B_TongDao
                4 C_ZhenLiaoShi
                5 JinQu
  overlays/   color overlays for inspection
  process_log.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

CLASS_NAMES = {
    0: "background",
    1: "P",
    2: "A_DaTing",
    3: "B_TongDao",
    4: "C_ZhenLiaoShi",
    5: "JinQu",
}

# OpenCV BGR colors, used only for visualization.
COLORS_BGR = {
    0: (35, 35, 35),       # background
    1: (255, 255, 255),    # P
    2: (255, 255, 0),      # A_DaTing
    3: (0, 165, 255),      # B_TongDao
    4: (255, 0, 255),      # C_ZhenLiaoShi
    5: (255, 0, 0),        # JinQu
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX semantic segmentation on BEV images."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="ONNX model path.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input BEV image file or directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root directory.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=320,
        help="Model input width and height. Default: 320.",
    )
    parser.add_argument(
        "--providers",
        default="CUDAExecutionProvider,CPUExecutionProvider",
        help=(
            "Comma-separated ONNX Runtime providers. "
            "Default: CUDAExecutionProvider,CPUExecutionProvider."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search input directories.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Process every Nth image. Default: 1.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Mask opacity in overlay. Default: 0.45.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not save overlay images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    return parser.parse_args()


def list_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()

    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def relative_image_path(image_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(image_path.name)
    return image_path.relative_to(input_path)


def choose_providers(requested: str) -> list[str]:
    available = ort.get_available_providers()
    requested_list = [
        item.strip()
        for item in requested.split(",")
        if item.strip()
    ]

    selected = [
        provider
        for provider in requested_list
        if provider in available
    ]

    if not selected:
        selected = ["CPUExecutionProvider"]

    return selected


def preprocess(image_bgr: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(
        image_bgr,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    tensor = np.ascontiguousarray(tensor)
    return tensor


def decode_output(raw_output: np.ndarray) -> np.ndarray:
    """
    Decode an ONNX output that already contains class IDs.

    Supported shapes:
      [1, H, W]
      [H, W]
      [1, 1, H, W]
    """
    output = np.asarray(raw_output)

    if output.ndim == 4 and output.shape[0] == 1 and output.shape[1] == 1:
        mask = output[0, 0]
    elif output.ndim == 3 and output.shape[0] == 1:
        mask = output[0]
    elif output.ndim == 2:
        mask = output
    else:
        raise RuntimeError(
            "Unexpected ONNX output shape. "
            f"Expected [1,H,W], [H,W], or [1,1,H,W], got {output.shape}."
        )

    if not np.issubdtype(mask.dtype, np.integer):
        rounded = np.rint(mask)
        if not np.allclose(mask, rounded, atol=1e-5):
            raise RuntimeError(
                f"ONNX output is not an integer class-ID mask. dtype={mask.dtype}"
            )
        mask = rounded

    mask = mask.astype(np.uint8)

    unique_ids = set(int(value) for value in np.unique(mask))
    valid_ids = set(CLASS_NAMES.keys())
    invalid_ids = sorted(unique_ids - valid_ids)

    if invalid_ids:
        raise RuntimeError(
            f"Unexpected class IDs in output: {invalid_ids}. "
            f"Expected only {sorted(valid_ids)}."
        )

    return mask


def colorize_mask(mask_id: np.ndarray) -> np.ndarray:
    height, width = mask_id.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, bgr in COLORS_BGR.items():
        color[mask_id == class_id] = bgr

    return color


def make_overlay(
    image_bgr: np.ndarray,
    mask_id: np.ndarray,
    alpha: float,
) -> np.ndarray:
    color = colorize_mask(mask_id)
    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        color,
        alpha,
        0.0,
    )

    y = 22
    for class_id, class_name in CLASS_NAMES.items():
        count = int(np.sum(mask_id == class_id))
        ratio = 100.0 * count / mask_id.size
        text = f"{class_id}: {class_name}  {ratio:.1f}%"

        cv2.putText(
            overlay,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 19

    return overlay


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1

    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        return 1

    if args.input_size <= 0:
        print("[ERROR] --input-size must be > 0", file=sys.stderr)
        return 1

    if args.every_n <= 0:
        print("[ERROR] --every-n must be > 0", file=sys.stderr)
        return 1

    if not 0.0 <= args.overlay_alpha <= 1.0:
        print("[ERROR] --overlay-alpha must be in [0, 1]", file=sys.stderr)
        return 1

    images = list_images(input_path, args.recursive)
    images = images[::args.every_n]

    if not images:
        print(f"[ERROR] No images found: {input_path}", file=sys.stderr)
        return 2

    masks_root = output_root / "masks_id"
    overlays_root = output_root / "overlays"

    masks_root.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        overlays_root.mkdir(parents=True, exist_ok=True)

    providers = choose_providers(args.providers)
    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if len(inputs) != 1:
        print(
            f"[ERROR] Expected one model input, got {len(inputs)}",
            file=sys.stderr,
        )
        return 3

    if len(outputs) < 1:
        print("[ERROR] Model has no outputs", file=sys.stderr)
        return 3

    input_name = inputs[0].name
    output_name = outputs[0].name

    print("=" * 80)
    print(f"Model       : {model_path}")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_root}")
    print(f"Images      : {len(images)}")
    print(f"Input tensor: {input_name}, shape={inputs[0].shape}, type={inputs[0].type}")
    print(f"Output      : {output_name}, shape={outputs[0].shape}, type={outputs[0].type}")
    print(f"Providers   : {session.get_providers()}")
    print("Class IDs:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}")
    print("=" * 80)

    rows: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    failed = 0

    for index, image_path in enumerate(images, start=1):
        relative_path = relative_image_path(image_path, input_path)
        mask_path = (masks_root / relative_path).with_suffix(".png")
        overlay_path = (overlays_root / relative_path).with_suffix(".jpg")

        if mask_path.exists() and not args.overwrite:
            skipped += 1
            rows.append({
                "image": str(relative_path),
                "status": "skipped",
                "width": "",
                "height": "",
                "mask_width": "",
                "mask_height": "",
                "class_ids": "",
                "reason": "mask exists",
            })
            continue

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("cv2.imread failed")

            original_height, original_width = image.shape[:2]
            tensor = preprocess(image, args.input_size)

            raw_outputs = session.run(
                [output_name],
                {input_name: tensor},
            )
            mask_id = decode_output(raw_outputs[0])

            if mask_id.shape != (original_height, original_width):
                mask_id = cv2.resize(
                    mask_id,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask_id):
                raise RuntimeError(f"Failed to write mask: {mask_path}")

            if not args.no_overlay:
                overlay = make_overlay(
                    image_bgr=image,
                    mask_id=mask_id,
                    alpha=args.overlay_alpha,
                )
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise RuntimeError(
                        f"Failed to write overlay: {overlay_path}"
                    )

            unique_ids = ",".join(
                str(int(value))
                for value in np.unique(mask_id)
            )

            rows.append({
                "image": str(relative_path),
                "status": "ok",
                "width": original_width,
                "height": original_height,
                "mask_width": mask_id.shape[1],
                "mask_height": mask_id.shape[0],
                "class_ids": unique_ids,
                "reason": "",
            })
            processed += 1

        except Exception as exc:
            failed += 1
            rows.append({
                "image": str(relative_path),
                "status": "failed",
                "width": "",
                "height": "",
                "mask_width": "",
                "mask_height": "",
                "class_ids": "",
                "reason": str(exc),
            })
            print(
                f"[ERROR] {image_path}: {exc}",
                file=sys.stderr,
            )

        if index == 1 or index % 100 == 0 or index == len(images):
            print(
                f"[{index}/{len(images)}] "
                f"processed={processed}, "
                f"skipped={skipped}, "
                f"failed={failed}"
            )

    log_path = output_root / "process_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "image",
            "status",
            "width",
            "height",
            "mask_width",
            "mask_height",
            "class_ids",
            "reason",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 80)
    print("Inference complete")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Masks     : {masks_root}")
    if not args.no_overlay:
        print(f"Overlays  : {overlays_root}")
    print(f"Log       : {log_path}")

    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
