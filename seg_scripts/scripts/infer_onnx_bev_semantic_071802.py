#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ONNX dense semantic-segmentation inference for BEV images.

Default model:
  /home/xhm/Desktop/relocate_ws/models/071802.onnx

Default input:
  /home/xhm/Desktop/relocate_ws/data/extracted/bev_all/2026-06-28_01-18-35

Default output:
  /home/xhm/Desktop/relocate_ws/data/extracted/071802onnxinfer03

Outputs:
  masks_id/      single-channel uint8 PNG class-ID masks
  overlays/      visualization overlays
  process_log.csv

Class IDs:
  0 JinQu
  1 C_ZhenLiaoShi
  2 B_TongDao
  3 A_DaTing
  4 P
  5 background
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_MODEL = Path("/home/xhm/Desktop/relocate_ws/models/071802.onnx")
DEFAULT_INPUT = Path(
    "/home/xhm/Desktop/relocate_ws/data/extracted/"
    "bev_all/2026-06-28_01-18-35"
)
DEFAULT_OUTPUT = Path(
    "/home/xhm/Desktop/relocate_ws/data/extracted/071802onnxinfer03"
)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

BACKGROUND_ID = 5
VALID_CLASS_IDS = set(CLASS_NAMES)

# OpenCV BGR colors used only for visualization.
COLORS_BGR = {
    0: (255, 0, 0),       # JinQu
    1: (255, 0, 255),     # C_ZhenLiaoShi
    2: (0, 165, 255),     # B_TongDao
    3: (255, 255, 0),     # A_DaTing
    4: (255, 255, 255),   # P
    5: (35, 35, 35),      # background
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ONNX semantic segmentation on BEV images."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input image file or directory. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT}",
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
        "--every-n",
        type=int,
        default=1,
        help="Process every Nth image after sorting. Default: 1.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Mask opacity in the overlay. Default: 0.45.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output masks.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not save visualization overlays.",
    )
    return parser.parse_args()


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator: Iterable[Path] = input_path.rglob("*")
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


def parse_model_names(metadata: dict[str, str]) -> dict[int, str] | None:
    raw_names = metadata.get("names")
    if not raw_names:
        return None

    try:
        parsed = ast.literal_eval(raw_names)
    except (ValueError, SyntaxError):
        return None

    if not isinstance(parsed, dict):
        return None

    try:
        return {int(key): str(value) for key, value in parsed.items()}
    except (TypeError, ValueError):
        return None


def preprocess(image_bgr: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(
        image_bgr,
        (input_size, input_size),
        interpolation=cv2.INTER_LINEAR,
    )
    image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = image_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return np.ascontiguousarray(tensor)


def decode_output(raw_output: np.ndarray) -> np.ndarray:
    """
    Supported ONNX outputs:

      [1, H, W]       direct class-ID mask
      [H, W]          direct class-ID mask
      [1, 1, H, W]    direct class-ID mask
      [1, 6, H, W]    class logits/probabilities; argmax on class axis
    """
    output = np.asarray(raw_output)

    if output.ndim == 4:
        if output.shape[0] != 1:
            raise RuntimeError(
                f"Only batch=1 is supported, got output shape {output.shape}"
            )

        if output.shape[1] == 1:
            mask = output[0, 0]
        elif output.shape[1] == len(CLASS_NAMES):
            mask = np.argmax(output, axis=1)[0]
        else:
            raise RuntimeError(
                "Unexpected 4D output shape. "
                f"Expected [1,1,H,W] or [1,6,H,W], got {output.shape}."
            )

    elif output.ndim == 3 and output.shape[0] == 1:
        mask = output[0]

    elif output.ndim == 2:
        mask = output

    else:
        raise RuntimeError(
            "Unexpected ONNX output shape. "
            f"Got {output.shape}."
        )

    if not np.issubdtype(mask.dtype, np.integer):
        rounded = np.rint(mask)
        if not np.allclose(mask, rounded, atol=1e-5):
            raise RuntimeError(
                "Decoded output is not an integer class-ID mask. "
                f"dtype={mask.dtype}"
            )
        mask = rounded

    mask = mask.astype(np.uint8)

    unique_ids = {int(value) for value in np.unique(mask)}
    invalid_ids = sorted(unique_ids - VALID_CLASS_IDS)
    if invalid_ids:
        raise RuntimeError(
            f"Output contains invalid class IDs {invalid_ids}; "
            f"expected only {sorted(VALID_CLASS_IDS)}."
        )

    return mask


def colorize_mask(mask_id: np.ndarray) -> np.ndarray:
    color = np.zeros(
        (mask_id.shape[0], mask_id.shape[1], 3),
        dtype=np.uint8,
    )

    for class_id, bgr in COLORS_BGR.items():
        color[mask_id == class_id] = bgr

    return color


def draw_legend(
    image: np.ndarray,
    mask_id: np.ndarray,
) -> np.ndarray:
    output = image.copy()
    total_pixels = mask_id.size

    y = 22
    for class_id, class_name in CLASS_NAMES.items():
        count = int(np.count_nonzero(mask_id == class_id))
        ratio = 100.0 * count / total_pixels
        text = f"{class_id}: {class_name}  {ratio:.1f}%"

        cv2.putText(
            output,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            text,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 19

    return output


def make_overlay(
    image_bgr: np.ndarray,
    mask_id: np.ndarray,
    alpha: float,
) -> np.ndarray:
    mask_color = colorize_mask(mask_id)

    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        mask_color,
        alpha,
        0.0,
    )
    overlay = draw_legend(overlay, mask_id)

    return overlay


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not model_path.is_file():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1

    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}", file=sys.stderr)
        return 1

    if args.input_size <= 0:
        print("[ERROR] --input-size must be > 0", file=sys.stderr)
        return 1

    if args.every_n <= 0:
        print("[ERROR] --every-n must be > 0", file=sys.stderr)
        return 1

    if not 0.0 <= args.overlay_alpha <= 1.0:
        print(
            "[ERROR] --overlay-alpha must be in [0, 1]",
            file=sys.stderr,
        )
        return 1

    try:
        all_images = list_images(input_path)
    except (OSError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    images = all_images[::args.every_n]
    if not images:
        print(f"[ERROR] No images found: {input_path}", file=sys.stderr)
        return 2

    masks_root = output_root / "masks_id"
    overlays_root = output_root / "overlays"

    masks_root.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        overlays_root.mkdir(parents=True, exist_ok=True)

    providers = choose_providers(args.providers)

    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=providers,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load ONNX model: {exc}", file=sys.stderr)
        return 3

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

    metadata = session.get_modelmeta().custom_metadata_map
    model_names = parse_model_names(metadata)

    print("=" * 88)
    print("ONNX BEV semantic inference")
    print(f"Model        : {model_path}")
    print(f"Input        : {input_path}")
    print(f"Output       : {output_root}")
    print(f"Images found : {len(all_images)}")
    print(f"Images used  : {len(images)}")
    print(f"Input tensor : {input_name}, {inputs[0].shape}, {inputs[0].type}")
    print(f"Output tensor: {output_name}, {outputs[0].shape}, {outputs[0].type}")
    print(f"Providers    : {session.get_providers()}")
    print("Class IDs:")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_id}: {class_name}")

    if model_names is not None:
        print(f"Model names  : {model_names}")
        if model_names != CLASS_NAMES:
            print(
                "[WARN] ONNX metadata names differ from the script mapping.\n"
                "       Numeric model output is not changed."
            )

    print("=" * 88)

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
                "class_ids": "",
                "foreground_ratio": "",
                "reason": "mask exists",
            })
            continue

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("cv2.imread failed")

            original_height, original_width = image.shape[:2]

            tensor = preprocess(
                image_bgr=image,
                input_size=args.input_size,
            )

            raw_output = session.run(
                [output_name],
                {input_name: tensor},
            )[0]

            mask_id = decode_output(raw_output)

            if mask_id.shape != (original_height, original_width):
                mask_id = cv2.resize(
                    mask_id,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask_id):
                raise RuntimeError(f"Failed to save mask: {mask_path}")

            if not args.no_overlay:
                overlay = make_overlay(
                    image_bgr=image,
                    mask_id=mask_id,
                    alpha=args.overlay_alpha,
                )
                overlay_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise RuntimeError(
                        f"Failed to save overlay: {overlay_path}"
                    )

            unique_ids = ",".join(
                str(int(value))
                for value in np.unique(mask_id)
            )
            foreground_ratio = float(
                np.count_nonzero(mask_id != BACKGROUND_ID)
            ) / mask_id.size

            rows.append({
                "image": str(relative_path),
                "status": "ok",
                "width": original_width,
                "height": original_height,
                "class_ids": unique_ids,
                "foreground_ratio": f"{foreground_ratio:.8f}",
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
                "class_ids": "",
                "foreground_ratio": "",
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
            "class_ids",
            "foreground_ratio",
            "reason",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=" * 88)
    print("Inference complete")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Masks     : {masks_root}")
    if not args.no_overlay:
        print(f"Overlays  : {overlays_root}")
    print(f"Log       : {log_path}")
    print("=" * 88)

    return 0 if failed == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
