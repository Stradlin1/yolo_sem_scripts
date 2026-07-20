#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use a trained YOLO26 semantic-segmentation checkpoint to infer every 50th
BEV image.

Fixed paths:
    model:
      /home/xhm/Desktop/sem_train/runs/
      verify_2026-06-28_01-08-45/weights/best.pt

    input:
      /home/xhm/Desktop/sem_train/datasets/
      bev_all/2026-06-28_01-05-54

    output:
      /home/xhm/Desktop/sem_train/datasets/test071801

Output layout:
    test071801/
    ├── masks_id/       Single-channel class-ID PNG masks
    ├── color_masks/    Colored masks for inspection
    ├── previews/       Original image + overlay + class legend
    └── process_log.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Fixed configuration
# ---------------------------------------------------------------------------

MODEL_PATH = Path(
    "/home/xhm/Desktop/sem_train/runs/"
    "verify_2026-06-28_01-08-45/weights/best.pt"
)

INPUT_DIR = Path(
    "/home/xhm/Desktop/sem_train/datasets/"
    "bev_all/2026-06-28_01-05-54"
)

OUTPUT_DIR = Path(
    "/home/xhm/Desktop/sem_train/datasets/test071801"
)

IMAGE_SIZE = 320
DEVICE = 0
EVERY_N = 50
OVERWRITE = True
OVERLAY_ALPHA = 0.45

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

# BGR colors used only for visualization.
COLORS_BGR = {
    0: (35, 35, 35),       # background
    1: (255, 255, 255),    # P
    2: (255, 255, 0),      # A_DaTing
    3: (0, 165, 255),      # B_TongDao
    4: (255, 0, 255),      # C_ZhenLiaoShi
    5: (255, 0, 0),        # JinQu
}


def list_images(root: Path) -> list[Path]:
    """Recursively list supported images in deterministic order."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def normalize_names(raw_names) -> dict[int, str]:
    """Convert Ultralytics model.names to a normal integer-keyed dictionary."""
    if isinstance(raw_names, dict):
        return {
            int(class_id): str(class_name)
            for class_id, class_name in raw_names.items()
        }

    if isinstance(raw_names, (list, tuple)):
        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(raw_names)
        }

    raise TypeError(f"Unsupported model.names type: {type(raw_names)}")


def tensor_to_mask(result) -> np.ndarray:
    """Extract the dense semantic class map from one Ultralytics result."""
    if result.semantic_mask is None:
        raise RuntimeError(
            "Result does not contain semantic_mask. "
            "Confirm that best.pt is a semantic-segmentation model."
        )

    data = result.semantic_mask.data

    if torch.is_tensor(data):
        mask = data.detach().cpu().numpy()
    else:
        mask = np.asarray(data)

    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise RuntimeError(
            f"Expected semantic mask shape (H,W), got {mask.shape}"
        )

    if not np.issubdtype(mask.dtype, np.integer):
        rounded = np.rint(mask)
        if not np.allclose(mask, rounded, atol=1e-5):
            raise RuntimeError(
                f"Semantic mask is not discrete class IDs: dtype={mask.dtype}"
            )
        mask = rounded

    return mask.astype(np.uint8)


def colorize_mask(
    mask_id: np.ndarray,
    class_names: dict[int, str],
) -> np.ndarray:
    """Convert a class-ID mask to a BGR visualization."""
    height, width = mask_id.shape
    color = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id in class_names:
        color_value = COLORS_BGR.get(
            class_id,
            (
                int((53 * class_id + 67) % 256),
                int((97 * class_id + 29) % 256),
                int((193 * class_id + 11) % 256),
            ),
        )
        color[mask_id == class_id] = color_value

    return color


def make_overlay(
    image_bgr: np.ndarray,
    color_mask: np.ndarray,
) -> np.ndarray:
    return cv2.addWeighted(
        image_bgr,
        1.0 - OVERLAY_ALPHA,
        color_mask,
        OVERLAY_ALPHA,
        0.0,
    )


def draw_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 30),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        output,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def draw_legend(
    mask_id: np.ndarray,
    class_names: dict[int, str],
    height: int,
    width: int = 420,
) -> np.ndarray:
    """Draw class colors and pixel percentages."""
    panel = np.full((height, width, 3), 35, dtype=np.uint8)
    total_pixels = mask_id.size

    cv2.putText(
        panel,
        "semantic prediction",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    unique_ids = sorted(int(value) for value in np.unique(mask_id))
    cv2.putText(
        panel,
        "pred IDs: " + ",".join(map(str, unique_ids)),
        (12, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (190, 225, 255),
        1,
        cv2.LINE_AA,
    )

    y = 82
    row_height = 38

    for class_id in sorted(class_names):
        class_name = class_names[class_id]
        color = COLORS_BGR.get(
            class_id,
            (
                int((53 * class_id + 67) % 256),
                int((97 * class_id + 29) % 256),
                int((193 * class_id + 11) % 256),
            ),
        )

        pixels = int(np.sum(mask_id == class_id))
        ratio = pixels / total_pixels * 100.0

        cv2.rectangle(
            panel,
            (12, y - 15),
            (40, y + 11),
            color,
            thickness=-1,
        )
        cv2.rectangle(
            panel,
            (12, y - 15),
            (40, y + 11),
            (255, 255, 255),
            thickness=1,
        )

        cv2.putText(
            panel,
            f"{class_id}: {class_name}",
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"{pixels} px  {ratio:.1f}%",
            (50, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        y += row_height

    return panel


def make_preview(
    image_bgr: np.ndarray,
    overlay: np.ndarray,
    mask_id: np.ndarray,
    class_names: dict[int, str],
    image_name: str,
) -> np.ndarray:
    original_panel = draw_title(
        image_bgr,
        f"original: {image_name}",
    )
    overlay_panel = draw_title(
        overlay,
        "semantic overlay",
    )
    legend_panel = draw_legend(
        mask_id,
        class_names,
        height=image_bgr.shape[0],
    )

    return np.hstack([
        original_panel,
        overlay_panel,
        legend_panel,
    ])


def check_paths() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested, but torch.cuda.is_available() is False."
        )


def main() -> int:
    try:
        check_paths()

        all_images = list_images(INPUT_DIR)
        selected_images = all_images[::EVERY_N]

        if not all_images:
            raise RuntimeError(f"No images found in: {INPUT_DIR}")

        if not selected_images:
            raise RuntimeError("No images selected for inference.")

        masks_dir = OUTPUT_DIR / "masks_id"
        color_masks_dir = OUTPUT_DIR / "color_masks"
        previews_dir = OUTPUT_DIR / "previews"

        masks_dir.mkdir(parents=True, exist_ok=True)
        color_masks_dir.mkdir(parents=True, exist_ok=True)
        previews_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 82)
        print("YOLO semantic inference")
        print(f"Ultralytics : {ultralytics.__version__}")
        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA        : {torch.cuda.is_available()}")
        if torch.cuda.is_available() and DEVICE != "cpu":
            print(f"GPU         : {torch.cuda.get_device_name(DEVICE)}")
        print(f"Model       : {MODEL_PATH}")
        print(f"Input       : {INPUT_DIR}")
        print(f"Output      : {OUTPUT_DIR}")
        print(f"All images  : {len(all_images)}")
        print(f"Every N     : {EVERY_N}")
        print(f"Selected    : {len(selected_images)}")
        print("=" * 82)

        model = YOLO(str(MODEL_PATH), task="semantic")
        class_names = normalize_names(model.names)

        print("Class mapping:")
        for class_id in sorted(class_names):
            print(f"  {class_id}: {class_names[class_id]}")
        print("=" * 82)

        rows: list[dict[str, object]] = []
        processed = 0
        skipped = 0
        failed = 0

        for selected_index, image_path in enumerate(selected_images):
            source_index = selected_index * EVERY_N
            relative_path = image_path.relative_to(INPUT_DIR)

            mask_path = (
                masks_dir / relative_path
            ).with_suffix(".png")
            color_mask_path = (
                color_masks_dir / relative_path
            ).with_suffix(".png")
            preview_path = (
                previews_dir / relative_path
            ).with_suffix(".jpg")

            if mask_path.exists() and not OVERWRITE:
                skipped += 1
                continue

            try:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError("cv2.imread failed")

                results = model.predict(
                    source=str(image_path),
                    imgsz=IMAGE_SIZE,
                    device=DEVICE,
                    verbose=False,
                    save=False,
                )

                if len(results) != 1:
                    raise RuntimeError(
                        f"Expected one result, got {len(results)}"
                    )

                mask_id = tensor_to_mask(results[0])

                image_height, image_width = image.shape[:2]
                if mask_id.shape != (image_height, image_width):
                    mask_id = cv2.resize(
                        mask_id,
                        (image_width, image_height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                unique_ids = sorted(
                    int(value) for value in np.unique(mask_id)
                )
                invalid_ids = sorted(
                    set(unique_ids) - set(class_names.keys())
                )
                if invalid_ids:
                    raise RuntimeError(
                        f"Prediction contains invalid IDs: {invalid_ids}"
                    )

                color_mask = colorize_mask(mask_id, class_names)
                overlay = make_overlay(image, color_mask)
                preview = make_preview(
                    image_bgr=image,
                    overlay=overlay,
                    mask_id=mask_id,
                    class_names=class_names,
                    image_name=image_path.name,
                )

                mask_path.parent.mkdir(parents=True, exist_ok=True)
                color_mask_path.parent.mkdir(parents=True, exist_ok=True)
                preview_path.parent.mkdir(parents=True, exist_ok=True)

                if not cv2.imwrite(str(mask_path), mask_id):
                    raise RuntimeError(f"Failed to save: {mask_path}")

                if not cv2.imwrite(str(color_mask_path), color_mask):
                    raise RuntimeError(
                        f"Failed to save: {color_mask_path}"
                    )

                if not cv2.imwrite(str(preview_path), preview):
                    raise RuntimeError(f"Failed to save: {preview_path}")

                rows.append({
                    "source_index": source_index,
                    "selected_index": selected_index,
                    "image": str(relative_path),
                    "width": image_width,
                    "height": image_height,
                    "class_ids": ",".join(map(str, unique_ids)),
                    "mask": str(mask_path),
                    "color_mask": str(color_mask_path),
                    "preview": str(preview_path),
                    "status": "ok",
                    "reason": "",
                })

                processed += 1

            except Exception as exc:
                failed += 1
                rows.append({
                    "source_index": source_index,
                    "selected_index": selected_index,
                    "image": str(relative_path),
                    "width": "",
                    "height": "",
                    "class_ids": "",
                    "mask": str(mask_path),
                    "color_mask": str(color_mask_path),
                    "preview": str(preview_path),
                    "status": "failed",
                    "reason": str(exc),
                })
                print(
                    f"[ERROR] {image_path}: {exc}",
                    file=sys.stderr,
                )

            print(
                f"[{selected_index + 1}/{len(selected_images)}] "
                f"source_index={source_index} "
                f"processed={processed} "
                f"failed={failed}"
            )

        log_path = OUTPUT_DIR / "process_log.csv"
        with log_path.open("w", newline="", encoding="utf-8") as file:
            fieldnames = [
                "source_index",
                "selected_index",
                "image",
                "width",
                "height",
                "class_ids",
                "mask",
                "color_mask",
                "preview",
                "status",
                "reason",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("=" * 82)
        print("Inference complete")
        print(f"Processed   : {processed}")
        print(f"Skipped     : {skipped}")
        print(f"Failed      : {failed}")
        print(f"Masks       : {masks_dir}")
        print(f"Color masks : {color_masks_dir}")
        print(f"Previews    : {previews_dir}")
        print(f"Log         : {log_path}")
        print("=" * 82)

        return 0 if failed == 0 else 2

    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
