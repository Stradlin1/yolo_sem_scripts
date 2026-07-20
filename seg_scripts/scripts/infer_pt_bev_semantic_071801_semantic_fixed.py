#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run YOLO26 semantic-segmentation inference on all BEV images.

This checkpoint is a semantic-segmentation model, not a YOLO instance-
segmentation model. Predictions are read from result.semantic_mask.

Default paths:
  model : /home/xhm/Desktop/relocate_ws/models/071801.pt
  input : /home/xhm/Desktop/relocate_ws/data/extracted/bev_all/2026-06-28_01-05-54
  output: /home/xhm/Desktop/relocate_ws/data/extracted/seg_pt/2026-06-28_01-05-54_071801

Output layout:
  masks_id/     single-channel uint8 semantic class-ID PNGs
  color_masks/  colored masks for checking
  overlays/     original image blended with the semantic mask
  process_log.csv

Class IDs:
  0 background
  1 P
  2 A_DaTing
  3 B_TongDao
  4 C_ZhenLiaoShi
  5 JinQu
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


DEFAULT_MODEL = Path("/home/xhm/Desktop/relocate_ws/models/071801.pt")
DEFAULT_INPUT = Path(
    "/home/xhm/Desktop/relocate_ws/data/extracted/bev_all/2026-06-28_01-05-54"
)
DEFAULT_OUTPUT = Path(
    "/home/xhm/Desktop/relocate_ws/data/extracted/seg_pt/"
    "2026-06-28_01-05-54_071801"
)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

EXPECTED_CLASS_NAMES = {
    0: "background",
    1: "P",
    2: "A_DaTing",
    3: "B_TongDao",
    4: "C_ZhenLiaoShi",
    5: "JinQu",
}

# OpenCV BGR colors. These affect only visualization files.
COLORS_BGR = {
    0: (35, 35, 35),
    1: (255, 255, 255),
    2: (255, 255, 0),
    3: (0, 165, 255),
    4: (255, 0, 255),
    5: (255, 0, 0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 071801.pt dense semantic-segmentation inference."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Process every Nth sorted image. Default: 1.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        help="Search input directories recursively. This is the default.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Search only the input directory itself.",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing masks and visualizations.",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Save masks_id only; do not save color masks or overlays.",
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(path))
    ]


def list_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file: {path}")
        return [path]

    iterator: Iterable[Path]
    iterator = path.rglob("*") if recursive else path.iterdir()

    return sorted(
        (
            item
            for item in iterator
            if item.is_file() and item.suffix.lower() in IMAGE_EXTS
        ),
        key=natural_key,
    )


def relative_image_path(image_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(image_path.name)
    return image_path.relative_to(input_path)


def normalize_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(class_id): str(name) for class_id, name in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(raw_names)}
    raise TypeError(f"Unsupported model.names type: {type(raw_names).__name__}")


def validate_model_names(class_names: dict[int, str]) -> None:
    expected_ids = set(EXPECTED_CLASS_NAMES)
    actual_ids = set(class_names)

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Model class IDs do not match the semantic dataset. "
            f"Expected {sorted(expected_ids)}, got {sorted(actual_ids)}."
        )

    mismatches = []
    for class_id in sorted(expected_ids):
        expected = EXPECTED_CLASS_NAMES[class_id].lower()
        actual = class_names[class_id].strip().lower()
        if actual != expected:
            mismatches.append(
                f"{class_id}: expected '{EXPECTED_CLASS_NAMES[class_id]}', "
                f"got '{class_names[class_id]}'"
            )

    if mismatches:
        raise RuntimeError(
            "Model class names do not match the training mask IDs:\n  "
            + "\n  ".join(mismatches)
        )


def semantic_result_to_mask(result, class_count: int) -> np.ndarray:
    """Extract a dense HxW class-ID map from a semantic result."""
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is None:
        raise RuntimeError(
            "Result has no semantic_mask. This usually means the checkpoint "
            "was loaded or inferred as the wrong task. The model must be "
            "loaded with task='semantic'."
        )

    data = getattr(semantic_mask, "data", semantic_mask)
    if torch.is_tensor(data):
        array = data.detach().cpu().numpy()
    else:
        array = np.asarray(data)

    array = np.squeeze(array)

    # Normal output from this semantic model is already a 2D class-ID map.
    if array.ndim == 2:
        if np.issubdtype(array.dtype, np.integer):
            mask = array
        else:
            rounded = np.rint(array)
            if not np.allclose(array, rounded, atol=1e-5):
                raise RuntimeError(
                    "2D semantic_mask is not a discrete class-ID map: "
                    f"shape={array.shape}, dtype={array.dtype}, "
                    f"range=({float(array.min()):.4f}, {float(array.max()):.4f})"
                )
            mask = rounded

    # Defensive fallback for versions that expose CxHxW logits/probabilities.
    elif array.ndim == 3 and array.shape[0] == class_count:
        mask = np.argmax(array, axis=0)
    elif array.ndim == 3 and array.shape[-1] == class_count:
        mask = np.argmax(array, axis=-1)
    else:
        raise RuntimeError(
            "Unexpected semantic_mask shape. Expected HxW class IDs or "
            f"CxHxW logits, got {array.shape}."
        )

    mask = mask.astype(np.uint8)
    unique_ids = sorted(int(value) for value in np.unique(mask))
    invalid_ids = sorted(set(unique_ids) - set(range(class_count)))
    if invalid_ids:
        raise RuntimeError(
            f"Prediction contains invalid class IDs {invalid_ids}; "
            f"valid IDs are 0..{class_count - 1}."
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
    color_mask: np.ndarray,
    mask_id: np.ndarray,
    alpha: float,
) -> np.ndarray:
    overlay = cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        color_mask,
        alpha,
        0.0,
    )

    total = mask_id.size
    y = 22
    for class_id, class_name in EXPECTED_CLASS_NAMES.items():
        pixels = int(np.count_nonzero(mask_id == class_id))
        ratio = 100.0 * pixels / total
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

    if not model_path.is_file():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1
    if not input_path.exists():
        print(f"[ERROR] Input path not found: {input_path}", file=sys.stderr)
        return 1
    if args.imgsz <= 0:
        print("[ERROR] --imgsz must be > 0", file=sys.stderr)
        return 1
    if args.every_n <= 0:
        print("[ERROR] --every-n must be > 0", file=sys.stderr)
        return 1
    if not 0.0 <= args.overlay_alpha <= 1.0:
        print("[ERROR] --overlay-alpha must be in [0, 1]", file=sys.stderr)
        return 1
    if args.device != "cpu" and not torch.cuda.is_available():
        print(
            "[ERROR] CUDA device requested but torch.cuda.is_available() is False.",
            file=sys.stderr,
        )
        return 1

    images = list_images(input_path, args.recursive)[::args.every_n]
    if not images:
        print(f"[ERROR] No images found: {input_path}", file=sys.stderr)
        return 2

    masks_root = output_root / "masks_id"
    color_masks_root = output_root / "color_masks"
    overlays_root = output_root / "overlays"

    masks_root.mkdir(parents=True, exist_ok=True)
    if not args.no_visuals:
        color_masks_root.mkdir(parents=True, exist_ok=True)
        overlays_root.mkdir(parents=True, exist_ok=True)

    # Critical: force the checkpoint to use the semantic task.
    model = YOLO(str(model_path), task="semantic")
    class_names = normalize_names(model.names)
    validate_model_names(class_names)

    print("=" * 88)
    print("YOLO dense semantic inference")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"Model task  : {model.task}")
    print(f"Model       : {model_path}")
    print(f"Input       : {input_path}")
    print(f"Output      : {output_root}")
    print(f"Images      : {len(images)}")
    print(f"imgsz/device: {args.imgsz} / {args.device}")
    print("Class IDs:")
    for class_id in sorted(class_names):
        print(f"  {class_id}: {class_names[class_id]}")
    print("=" * 88)

    rows: list[dict[str, object]] = []
    processed = 0
    skipped = 0
    failed = 0
    all_background_count = 0

    for index, image_path in enumerate(images, start=1):
        relative_path = relative_image_path(image_path, input_path)
        mask_path = (masks_root / relative_path).with_suffix(".png")
        color_mask_path = (color_masks_root / relative_path).with_suffix(".png")
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

            # Do not use result.masks/result.boxes here. This is dense semantic
            # segmentation and its output is result.semantic_mask.
            results = model.predict(
                source=str(image_path),
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
                save=False,
            )

            if len(results) != 1:
                raise RuntimeError(f"Expected one result, got {len(results)}")

            mask_id = semantic_result_to_mask(
                results[0],
                class_count=len(class_names),
            )

            image_height, image_width = image.shape[:2]
            if mask_id.shape != (image_height, image_width):
                mask_id = cv2.resize(
                    mask_id,
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            unique_ids = sorted(int(value) for value in np.unique(mask_id))
            foreground_ratio = float(np.count_nonzero(mask_id)) / mask_id.size
            if foreground_ratio == 0.0:
                all_background_count += 1

            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), mask_id):
                raise RuntimeError(f"Failed to save mask: {mask_path}")

            if not args.no_visuals:
                color_mask = colorize_mask(mask_id)
                overlay = make_overlay(
                    image_bgr=image,
                    color_mask=color_mask,
                    mask_id=mask_id,
                    alpha=args.overlay_alpha,
                )

                color_mask_path.parent.mkdir(parents=True, exist_ok=True)
                overlay_path.parent.mkdir(parents=True, exist_ok=True)

                if not cv2.imwrite(str(color_mask_path), color_mask):
                    raise RuntimeError(
                        f"Failed to save color mask: {color_mask_path}"
                    )
                if not cv2.imwrite(str(overlay_path), overlay):
                    raise RuntimeError(f"Failed to save overlay: {overlay_path}")

            rows.append({
                "image": str(relative_path),
                "status": "ok",
                "width": image_width,
                "height": image_height,
                "class_ids": ",".join(map(str, unique_ids)),
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
            print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)

        if index == 1 or index % 25 == 0 or index == len(images):
            print(
                f"[{index}/{len(images)}] processed={processed}, "
                f"skipped={skipped}, failed={failed}, "
                f"all_background={all_background_count}"
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
    print("Semantic inference complete")
    print(f"Processed      : {processed}")
    print(f"Skipped        : {skipped}")
    print(f"Failed         : {failed}")
    print(f"All-background : {all_background_count}")
    print(f"Masks          : {masks_root}")
    if not args.no_visuals:
        print(f"Color masks    : {color_masks_root}")
        print(f"Overlays       : {overlays_root}")
    print(f"Log            : {log_path}")
    print("=" * 88)

    if processed > 0 and all_background_count == processed:
        print(
            "[WARN] Every generated mask is background-only. The script is now "
            "reading semantic_mask correctly, so this would indicate a model "
            "or dataset problem rather than an instance/semantic API mismatch.",
            file=sys.stderr,
        )

    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
