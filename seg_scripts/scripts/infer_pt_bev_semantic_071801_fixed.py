#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Ultralytics YOLO segmentation inference on BEV images and convert the
instance-segmentation output into the same semantic-mask format used by the
previous ONNX workflow.

Default inputs:
  model : /home/xhm/Desktop/relocate_ws/models/071801.pt
  images: /home/xhm/Desktop/relocate_ws/data/extracted/bev_all/2026-06-28_01-05-54

Default output:
  /home/xhm/Desktop/relocate_ws/data/extracted/seg_pt/2026-06-28_01-05-54_071801
    masks_id/       single-channel uint8 PNG semantic labels
    overlays/       visualization images
    process_log.csv

Semantic mask pixel IDs:
  0 background
  1 P
  2 A_DaTing
  3 B_TongDao
  4 C_ZhenLiaoShi
  5 JinQu

The PT model may use either of these class layouts:
  - six classes: 0=background, 1=P, ..., 5=JinQu
  - five classes: 0=P, ..., 4=JinQu

Class names are used to build the mapping automatically. Semantic output always
uses pixel IDs 0..5, with 0 reserved for background.
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

CLASS_NAMES = {
    0: "background",
    1: "P",
    2: "A_DaTing",
    3: "B_TongDao",
    4: "C_ZhenLiaoShi",
    5: "JinQu",
}

COLORS_BGR = {
    0: (35, 35, 35),
    1: (255, 255, 255),
    2: (255, 255, 0),
    3: (0, 165, 255),
    4: (255, 0, 255),
    5: (255, 0, 0),
}

EXPECTED_NAME_TO_SEMANTIC_ID = {
    "background": 0,
    "p": 1,
    "a_dating": 2,
    "b_tongdao": 3,
    "c_zhenliaoshi": 4,
    "jinqu": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run 071801.pt YOLO-seg inference and save semantic class-ID masks."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"PT model path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input BEV image file or directory. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output root directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="YOLO inference size. Default: 320.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.70,
        help="NMS IoU threshold. Default: 0.70.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.50,
        help="Threshold applied to predicted masks. Default: 0.50.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device, for example 0 or cpu. Default: 0.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Inference batch size. Default: 16.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Ultralytics dataloader workers. Default: 4.",
    )
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
        help="Recursively search the input directory. This is the default.",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Search only the input directory itself.",
    )
    parser.set_defaults(recursive=True)
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="Mask opacity in overlay images. Default: 0.45.",
    )
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not save overlay images.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference. Enable only on a compatible CUDA device.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing semantic masks.",
    )
    parser.add_argument(
        "--allow-index-fallback",
        action="store_true",
        help=(
            "If PT class names do not match, use numeric fallback: "
            "0..5 -> semantic 0..5, or 0..4 -> semantic 1..5."
        ),
    )
    return parser.parse_args()


def natural_key(path: Path) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", str(path))
    ]


def list_images(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image file: {input_path}")
        return [input_path]

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    return sorted(images, key=natural_key)


def relative_image_path(image_path: Path, input_path: Path) -> Path:
    if input_path.is_file():
        return Path(image_path.name)
    return image_path.relative_to(input_path)


def normalize_class_name(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text


def normalize_model_names(names: object) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}
    raise RuntimeError(f"Unsupported model.names type: {type(names).__name__}")


def build_class_mapping(
    model_names: dict[int, str],
    allow_index_fallback: bool,
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    unknown: list[tuple[int, str]] = []

    for model_class_id, class_name in sorted(model_names.items()):
        normalized = normalize_class_name(class_name)
        semantic_id = EXPECTED_NAME_TO_SEMANTIC_ID.get(normalized)
        if semantic_id is None:
            unknown.append((model_class_id, class_name))
        else:
            mapping[model_class_id] = semantic_id

    mapped_semantic_ids = set(mapping.values())
    valid_named_layouts = {
        frozenset(range(6)),      # background + five foreground classes
        frozenset(range(1, 6)),  # five foreground classes; background is implicit
    }

    if not unknown and frozenset(mapped_semantic_ids) in valid_named_layouts:
        return mapping

    model_ids = set(model_names.keys())
    if allow_index_fallback and model_ids == set(range(6)):
        print(
            "[WARN] Model class names do not exactly match the expected names. "
            "Using direct index mapping 0..5 -> semantic IDs 0..5."
        )
        return {class_id: class_id for class_id in range(6)}

    if allow_index_fallback and model_ids == set(range(5)):
        print(
            "[WARN] Model class names do not exactly match the expected names. "
            "Using index mapping 0..4 -> semantic IDs 1..5."
        )
        return {class_id: class_id + 1 for class_id in range(5)}

    details = ", ".join(
        f"{class_id}:{name}" for class_id, name in sorted(model_names.items())
    )
    raise RuntimeError(
        "PT model class names do not match a supported class layout. "
        f"Model names: {details}. Supported layouts are either "
        "0:background, 1:P, 2:A_DaTing, 3:B_TongDao, "
        "4:C_ZhenLiaoShi, 5:JinQu; or five foreground classes "
        "P, A_DaTing, B_TongDao, C_ZhenLiaoShi, JinQu. "
        "If the numeric order is definitely correct, rerun with "
        "--allow-index-fallback."
    )


def validate_mask_ids(mask: np.ndarray) -> None:
    valid_ids = set(CLASS_NAMES.keys())
    unique_ids = set(int(value) for value in np.unique(mask))
    invalid = sorted(unique_ids - valid_ids)
    if invalid:
        raise RuntimeError(
            f"Generated semantic mask contains invalid IDs: {invalid}"
        )


def result_to_semantic_mask(
    result,
    image_height: int,
    image_width: int,
    class_mapping: dict[int, int],
    mask_threshold: float,
) -> tuple[np.ndarray, int]:
    semantic = np.zeros((image_height, image_width), dtype=np.uint8)

    if result.masks is None or result.boxes is None:
        return semantic, 0

    mask_tensor = result.masks.data
    class_tensor = result.boxes.cls
    confidence_tensor = result.boxes.conf

    masks = mask_tensor.detach().float().cpu().numpy()
    class_ids = class_tensor.detach().cpu().numpy().astype(np.int64)
    confidences = confidence_tensor.detach().float().cpu().numpy()

    if masks.shape[0] != class_ids.shape[0] or masks.shape[0] != confidences.shape[0]:
        raise RuntimeError(
            "YOLO result mismatch: number of masks, classes, and confidences differs."
        )

    score_map = np.full((image_height, image_width), -1.0, dtype=np.float32)

    for mask, model_class_id, confidence in zip(
        masks, class_ids, confidences
    ):
        model_class_id = int(model_class_id)
        if model_class_id not in class_mapping:
            raise RuntimeError(
                f"Prediction contains unmapped model class ID {model_class_id}."
            )

        if mask.shape != (image_height, image_width):
            mask = cv2.resize(
                mask,
                (image_width, image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        binary = mask >= mask_threshold
        update = binary & (float(confidence) > score_map)
        semantic[update] = class_mapping[model_class_id]
        score_map[update] = float(confidence)

    validate_mask_ids(semantic)
    return semantic, int(masks.shape[0])


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


def write_log(log_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "image",
        "status",
        "width",
        "height",
        "detections",
        "semantic_ids",
        "reason",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    if args.imgsz <= 0 or args.batch <= 0 or args.workers < 0:
        print("[ERROR] imgsz and batch must be > 0; workers must be >= 0.", file=sys.stderr)
        return 1
    if args.every_n <= 0:
        print("[ERROR] --every-n must be > 0.", file=sys.stderr)
        return 1
    if not 0.0 <= args.conf <= 1.0:
        print("[ERROR] --conf must be in [0, 1].", file=sys.stderr)
        return 1
    if not 0.0 <= args.iou <= 1.0:
        print("[ERROR] --iou must be in [0, 1].", file=sys.stderr)
        return 1
    if not 0.0 <= args.mask_threshold <= 1.0:
        print("[ERROR] --mask-threshold must be in [0, 1].", file=sys.stderr)
        return 1
    if not 0.0 <= args.overlay_alpha <= 1.0:
        print("[ERROR] --overlay-alpha must be in [0, 1].", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "[ERROR] ultralytics is not installed in the active environment. "
            "Activate the YOLO environment first.",
            file=sys.stderr,
        )
        return 2

    images = list_images(input_path, args.recursive)[::args.every_n]
    if not images:
        print(f"[ERROR] No images found: {input_path}", file=sys.stderr)
        return 2

    masks_root = output_root / "masks_id"
    overlays_root = output_root / "overlays"
    masks_root.mkdir(parents=True, exist_ok=True)
    if not args.no_overlay:
        overlays_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    model_names = normalize_model_names(model.names)
    class_mapping = build_class_mapping(
        model_names,
        allow_index_fallback=args.allow_index_fallback,
    )

    pending: list[Path] = []
    rows: list[dict[str, object]] = []
    skipped = 0

    for image_path in images:
        relative_path = relative_image_path(image_path, input_path)
        mask_path = (masks_root / relative_path).with_suffix(".png")
        if mask_path.exists() and not args.overwrite:
            skipped += 1
            rows.append({
                "image": str(relative_path),
                "status": "skipped",
                "width": "",
                "height": "",
                "detections": "",
                "semantic_ids": "",
                "reason": "mask exists",
            })
        else:
            pending.append(image_path)

    print("=" * 88)
    print(f"Model        : {model_path}")
    print(f"Input        : {input_path}")
    print(f"Output       : {output_root}")
    print(f"Found images : {len(images)}")
    print(f"Pending      : {len(pending)}")
    print(f"Skipped      : {skipped}")
    print(f"imgsz/conf   : {args.imgsz} / {args.conf}")
    print(f"device/batch : {args.device} / {args.batch}")
    print("Model class -> semantic mask mapping:")
    for model_class_id, semantic_id in sorted(class_mapping.items()):
        print(
            f"  model {model_class_id}:{model_names[model_class_id]} "
            f"-> mask {semantic_id}:{CLASS_NAMES[semantic_id]}"
        )
    print("=" * 88)

    processed = 0
    failed = 0

    for batch_start in range(0, len(pending), args.batch):
        batch_paths = pending[batch_start:batch_start + args.batch]

        try:
            results = model.predict(
                source=[str(path) for path in batch_paths],
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                batch=len(batch_paths),
                workers=args.workers,
                retina_masks=True,
                half=args.half,
                verbose=False,
                save=False,
            )
        except Exception as exc:
            for image_path in batch_paths:
                relative_path = relative_image_path(image_path, input_path)
                failed += 1
                rows.append({
                    "image": str(relative_path),
                    "status": "failed",
                    "width": "",
                    "height": "",
                    "detections": "",
                    "semantic_ids": "",
                    "reason": f"batch inference failed: {exc}",
                })
                print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)
            continue

        if len(results) != len(batch_paths):
            print(
                "[ERROR] Ultralytics returned a different number of results "
                "than input images.",
                file=sys.stderr,
            )
            return 4

        for image_path, result in zip(batch_paths, results):
            relative_path = relative_image_path(image_path, input_path)
            mask_path = (masks_root / relative_path).with_suffix(".png")
            overlay_path = (overlays_root / relative_path).with_suffix(".jpg")

            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("cv2.imread failed")

                image_height, image_width = image.shape[:2]
                semantic_mask, detections = result_to_semantic_mask(
                    result=result,
                    image_height=image_height,
                    image_width=image_width,
                    class_mapping=class_mapping,
                    mask_threshold=args.mask_threshold,
                )

                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(mask_path), semantic_mask):
                    raise RuntimeError(f"Failed to write mask: {mask_path}")

                if not args.no_overlay:
                    overlay = make_overlay(
                        image_bgr=image,
                        mask_id=semantic_mask,
                        alpha=args.overlay_alpha,
                    )
                    overlay_path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(overlay_path), overlay):
                        raise RuntimeError(
                            f"Failed to write overlay: {overlay_path}"
                        )

                semantic_ids = ",".join(
                    str(int(value)) for value in np.unique(semantic_mask)
                )
                rows.append({
                    "image": str(relative_path),
                    "status": "ok",
                    "width": image_width,
                    "height": image_height,
                    "detections": detections,
                    "semantic_ids": semantic_ids,
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
                    "detections": "",
                    "semantic_ids": "",
                    "reason": str(exc),
                })
                print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)

        done = min(batch_start + len(batch_paths), len(pending))
        print(
            f"[{done}/{len(pending)} pending] "
            f"processed={processed}, skipped={skipped}, failed={failed}"
        )

    log_path = output_root / "process_log.csv"
    write_log(log_path, rows)

    print("=" * 88)
    print("PT semantic inference complete")
    print(f"Processed : {processed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Masks     : {masks_root}")
    if not args.no_overlay:
        print(f"Overlays  : {overlays_root}")
    print(f"Log       : {log_path}")

    return 0 if failed == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
