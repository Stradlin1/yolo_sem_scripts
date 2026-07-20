#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
在独立测试集上运行 BEV 语义分割 ONNX，并计算 6 类语义分割指标。

默认路径：
    ONNX:
        /home/xhm/Desktop/onnx_transfer/best.onnx

    测试图片:
        /home/xhm/Desktop/onnx_transfer/
        bev_sem_round2_merged/images/test

    测试标签:
        /home/xhm/Desktop/onnx_transfer/
        bev_sem_round2_merged/masks/test

    输出:
        /home/xhm/Desktop/onnx_transfer/onnx_test_results

类别映射：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background

输出内容：
    predictions/       ONNX 预测的单通道类别 ID 图
    overlays/          原图叠加预测结果
    comparisons/       原图、GT、预测三联图
    metrics.csv        每类 IoU、准确率等
    confusion_matrix.csv
    per_image_metrics.csv
    summary.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_ROOT = Path("/home/xhm/Desktop/onnx_transfer")
DEFAULT_MODEL = DEFAULT_ROOT / "best.onnx"
DEFAULT_IMAGES = (
    DEFAULT_ROOT / "bev_sem_round2_merged" / "images" / "test"
)
DEFAULT_MASKS = (
    DEFAULT_ROOT / "bev_sem_round2_merged" / "masks" / "test"
)
DEFAULT_OUTPUT = DEFAULT_ROOT / "onnx_test_results"

CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

NUM_CLASSES = len(CLASS_NAMES)
VALID_CLASS_IDS = set(CLASS_NAMES)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

# 仅用于可视化，不影响类别数字。
CLASS_COLORS_BGR = {
    0: (0, 0, 255),       # JinQu
    1: (0, 165, 255),     # C_ZhenLiaoShi
    2: (0, 255, 255),     # B_TongDao
    3: (0, 255, 0),       # A_DaTing
    4: (255, 0, 0),       # P
    5: (80, 80, 80),      # background
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BEV semantic ONNX on the independent test set."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX 模型路径。默认: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help=f"测试图片根目录。默认: {DEFAULT_IMAGES}",
    )
    parser.add_argument(
        "--masks",
        type=Path,
        default=DEFAULT_MASKS,
        help=f"测试标签根目录。默认: {DEFAULT_MASKS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出目录。默认: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="预测 overlay 透明度。默认: 0.45",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已有输出。",
    )
    return parser.parse_args()


def natural_key(path: Path) -> tuple:
    import re

    text = path.as_posix().lower()
    parts = re.split(r"(\d+)", text)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def list_images(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda path: natural_key(path.relative_to(root)),
    )


def get_providers() -> list[str]:
    available = ort.get_available_providers()

    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    return providers


def get_static_dim(dim: object) -> int | None:
    if isinstance(dim, int) and dim > 0:
        return dim
    return None


def load_mask(mask_path: Path) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f"无法读取标签: {mask_path}")

    if mask.ndim == 3:
        channels_equal = (
            np.array_equal(mask[:, :, 0], mask[:, :, 1])
            and np.array_equal(mask[:, :, 1], mask[:, :, 2])
        )
        if not channels_equal:
            raise RuntimeError(
                f"标签不是单通道类别图: {mask_path}, shape={mask.shape}"
            )
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise RuntimeError(
            f"标签维度错误: {mask_path}, shape={mask.shape}"
        )

    mask = mask.astype(np.int64, copy=False)

    invalid = sorted(set(np.unique(mask).tolist()) - VALID_CLASS_IDS)
    if invalid:
        raise RuntimeError(
            f"标签包含非法类别值: {mask_path}, invalid={invalid}"
        )

    return mask


def preprocess(
    image_bgr: np.ndarray,
    input_h: int,
    input_w: int,
) -> np.ndarray:
    resized = cv2.resize(
        image_bgr,
        (input_w, input_h),
        interpolation=cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor)


def output_to_mask(
    output: np.ndarray,
    expected_h: int,
    expected_w: int,
) -> np.ndarray:
    output = np.asarray(output)

    # 当前模型协议：直接输出 [1, H, W] 类别 ID。
    if output.ndim == 3 and output.shape[0] == 1:
        mask = output[0]

    # 兼容 [H, W]。
    elif output.ndim == 2:
        mask = output

    # 兼容 logits [1, C, H, W]。
    elif output.ndim == 4 and output.shape[0] == 1:
        if output.shape[1] == NUM_CLASSES:
            mask = np.argmax(output[0], axis=0)
        elif output.shape[-1] == NUM_CLASSES:
            mask = np.argmax(output[0], axis=-1)
        else:
            raise RuntimeError(
                f"无法识别四维输出形状: {output.shape}"
            )

    # 兼容 logits [C, H, W]。
    elif output.ndim == 3 and output.shape[0] == NUM_CLASSES:
        mask = np.argmax(output, axis=0)

    else:
        raise RuntimeError(
            f"无法识别 ONNX 输出形状: {output.shape}"
        )

    if mask.shape != (expected_h, expected_w):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (expected_w, expected_h),
            interpolation=cv2.INTER_NEAREST,
        )

    mask = mask.astype(np.uint8, copy=False)

    invalid = sorted(set(np.unique(mask).tolist()) - VALID_CLASS_IDS)
    if invalid:
        raise RuntimeError(
            f"ONNX 输出包含非法类别值: {invalid}"
        )

    return mask


def colorize(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, bgr in CLASS_COLORS_BGR.items():
        color[mask == class_id] = bgr

    return color


def make_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    color_mask = colorize(mask)
    return cv2.addWeighted(
        image_bgr,
        1.0 - alpha,
        color_mask,
        alpha,
        0.0,
    )


def put_title(image: np.ndarray, title: str) -> np.ndarray:
    output = image.copy()
    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 34),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(
        output,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def update_confusion_matrix(
    confusion: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
) -> None:
    valid = (
        (gt >= 0)
        & (gt < NUM_CLASSES)
        & (pred >= 0)
        & (pred < NUM_CLASSES)
    )

    encoded = NUM_CLASSES * gt[valid] + pred[valid]
    counts = np.bincount(
        encoded,
        minlength=NUM_CLASSES * NUM_CLASSES,
    )
    confusion += counts.reshape(NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(
    confusion: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []

    tp = np.diag(confusion).astype(np.float64)
    gt_pixels = confusion.sum(axis=1).astype(np.float64)
    pred_pixels = confusion.sum(axis=0).astype(np.float64)
    union = gt_pixels + pred_pixels - tp

    iou = np.divide(
        tp,
        union,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=union > 0,
    )
    recall = np.divide(
        tp,
        gt_pixels,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=gt_pixels > 0,
    )
    precision = np.divide(
        tp,
        pred_pixels,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=pred_pixels > 0,
    )

    total_pixels = float(confusion.sum())
    correct_pixels = float(tp.sum())
    pixel_accuracy = (
        correct_pixels / total_pixels
        if total_pixels > 0
        else float("nan")
    )

    valid_iou = iou[~np.isnan(iou)]
    mean_iou = (
        float(valid_iou.mean())
        if len(valid_iou)
        else float("nan")
    )

    for class_id, class_name in CLASS_NAMES.items():
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "gt_pixels": int(gt_pixels[class_id]),
            "pred_pixels": int(pred_pixels[class_id]),
            "true_positive_pixels": int(tp[class_id]),
            "iou": float(iou[class_id]),
            "precision": float(precision[class_id]),
            "recall": float(recall[class_id]),
        })

    summary = {
        "mIoU": mean_iou,
        "PixelAccuracy": pixel_accuracy,
        "TotalPixels": total_pixels,
        "CorrectPixels": correct_pixels,
    }

    return rows, summary


def safe_float(value: float) -> str:
    return "nan" if np.isnan(value) else f"{value:.8f}"


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    image_root = args.images.expanduser().resolve()
    mask_root = args.masks.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not model_path.is_file():
        print(f"[ERROR] ONNX 不存在: {model_path}", file=sys.stderr)
        return 1

    if not image_root.is_dir():
        print(f"[ERROR] 测试图片目录不存在: {image_root}", file=sys.stderr)
        return 1

    if not mask_root.is_dir():
        print(f"[ERROR] 测试标签目录不存在: {mask_root}", file=sys.stderr)
        return 1

    if output_root.exists() and not args.overwrite:
        print(
            f"[ERROR] 输出目录已存在: {output_root}\n"
            "确认后添加 --overwrite。",
            file=sys.stderr,
        )
        return 1

    prediction_root = output_root / "predictions"
    overlay_root = output_root / "overlays"
    comparison_root = output_root / "comparisons"

    prediction_root.mkdir(parents=True, exist_ok=True)
    overlay_root.mkdir(parents=True, exist_ok=True)
    comparison_root.mkdir(parents=True, exist_ok=True)

    providers = get_providers()
    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_name = input_meta.name
    output_name = output_meta.name
    input_shape = input_meta.shape

    input_h = get_static_dim(input_shape[-2]) or 320
    input_w = get_static_dim(input_shape[-1]) or 320

    images = list_images(image_root)
    if not images:
        print(f"[ERROR] 没有找到测试图片: {image_root}", file=sys.stderr)
        return 1

    print("=" * 96)
    print("BEV semantic ONNX test evaluation")
    print(f"Model      : {model_path}")
    print(f"Images     : {image_root}")
    print(f"Masks      : {mask_root}")
    print(f"Output     : {output_root}")
    print(f"Providers  : {session.get_providers()}")
    print(f"Input      : {input_name}, shape={input_shape}")
    print(f"Output     : {output_name}, shape={output_meta.shape}")
    print(f"Images     : {len(images)}")
    print("=" * 96)

    confusion = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    per_image_rows: list[dict[str, object]] = []
    inference_times_ms: list[float] = []

    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(image_root)
        gt_path = (mask_root / relative_path).with_suffix(".png")

        if not gt_path.is_file():
            raise RuntimeError(
                "测试图片缺少对应标签:\n"
                f"  image: {image_path}\n"
                f"  mask : {gt_path}"
            )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {image_path}")

        gt = load_mask(gt_path)

        image_h, image_w = image.shape[:2]
        if gt.shape != (image_h, image_w):
            raise RuntimeError(
                "测试图片与标签尺寸不一致:\n"
                f"  image: {image_path} -> {image_w}x{image_h}\n"
                f"  mask : {gt_path} -> {gt.shape[1]}x{gt.shape[0]}"
            )

        tensor = preprocess(
            image_bgr=image,
            input_h=input_h,
            input_w=input_w,
        )

        start_time = time.perf_counter()
        outputs = session.run([output_name], {input_name: tensor})
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        inference_times_ms.append(elapsed_ms)

        pred = output_to_mask(
            outputs[0],
            expected_h=image_h,
            expected_w=image_w,
        )

        image_confusion = np.zeros_like(confusion)
        update_confusion_matrix(image_confusion, gt, pred)
        confusion += image_confusion

        _, image_summary = calculate_metrics(image_confusion)

        pred_path = (
            prediction_root / relative_path
        ).with_suffix(".png")
        overlay_path = (
            overlay_root / relative_path
        ).with_suffix(".png")
        comparison_path = (
            comparison_root / relative_path
        ).with_suffix(".jpg")

        pred_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(pred_path), pred)

        pred_overlay = make_overlay(image, pred, args.alpha)
        cv2.imwrite(str(overlay_path), pred_overlay)

        gt_overlay = make_overlay(image, gt, args.alpha)

        comparison = np.hstack([
            put_title(image, "Image"),
            put_title(gt_overlay, "Ground Truth"),
            put_title(pred_overlay, "ONNX Prediction"),
        ])
        cv2.imwrite(str(comparison_path), comparison)

        per_image_rows.append({
            "relative_path": relative_path.as_posix(),
            "mIoU": safe_float(image_summary["mIoU"]),
            "PixelAccuracy": safe_float(
                image_summary["PixelAccuracy"]
            ),
            "inference_ms": f"{elapsed_ms:.4f}",
        })

        if index == 1 or index % 100 == 0 or index == len(images):
            print(
                f"[{index}/{len(images)}] "
                f"{relative_path.as_posix()} | "
                f"{elapsed_ms:.2f} ms"
            )

    class_rows, summary = calculate_metrics(confusion)

    metrics_path = output_root / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "class_id",
            "class_name",
            "gt_pixels",
            "pred_pixels",
            "true_positive_pixels",
            "iou",
            "precision",
            "recall",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in class_rows:
            output_row = dict(row)
            for key in ("iou", "precision", "recall"):
                output_row[key] = safe_float(float(output_row[key]))
            writer.writerow(output_row)

    confusion_path = output_root / "confusion_matrix.csv"
    with confusion_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["gt\\pred"]
            + [
                f"{class_id}_{CLASS_NAMES[class_id]}"
                for class_id in range(NUM_CLASSES)
            ]
        )

        for class_id in range(NUM_CLASSES):
            writer.writerow(
                [f"{class_id}_{CLASS_NAMES[class_id]}"]
                + confusion[class_id].tolist()
            )

    per_image_path = output_root / "per_image_metrics.csv"
    with per_image_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "relative_path",
                "mIoU",
                "PixelAccuracy",
                "inference_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image_rows)

    mean_inference_ms = float(np.mean(inference_times_ms))
    median_inference_ms = float(np.median(inference_times_ms))
    fps = 1000.0 / mean_inference_ms if mean_inference_ms > 0 else 0.0

    summary_path = output_root / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("BEV semantic ONNX independent test evaluation\n")
        file.write("=" * 72 + "\n")
        file.write(f"Model: {model_path}\n")
        file.write(f"Images: {len(images)}\n")
        file.write(f"mIoU: {summary['mIoU']:.8f}\n")
        file.write(
            f"PixelAccuracy: {summary['PixelAccuracy']:.8f}\n"
        )
        file.write(
            f"Mean inference time: {mean_inference_ms:.4f} ms\n"
        )
        file.write(
            f"Median inference time: {median_inference_ms:.4f} ms\n"
        )
        file.write(f"Approx FPS: {fps:.4f}\n")
        file.write("\nPer-class metrics\n")
        file.write("-" * 72 + "\n")

        for row in class_rows:
            file.write(
                f"{row['class_id']} {row['class_name']}: "
                f"IoU={safe_float(float(row['iou']))}, "
                f"Precision={safe_float(float(row['precision']))}, "
                f"Recall={safe_float(float(row['recall']))}\n"
            )

    print("\nIndependent test result")
    print("=" * 96)
    print(f"Images        : {len(images)}")
    print(f"mIoU          : {summary['mIoU']:.6f}")
    print(f"PixelAccuracy : {summary['PixelAccuracy']:.6f}")
    print(f"Mean time     : {mean_inference_ms:.3f} ms/image")
    print(f"Approx FPS    : {fps:.2f}")
    print("-" * 96)

    for row in class_rows:
        print(
            f"{row['class_id']} {row['class_name']:<18} "
            f"IoU={safe_float(float(row['iou']))}  "
            f"Precision={safe_float(float(row['precision']))}  "
            f"Recall={safe_float(float(row['recall']))}"
        )

    print("-" * 96)
    print(f"Predictions   : {prediction_root}")
    print(f"Overlays      : {overlay_root}")
    print(f"Comparisons   : {comparison_root}")
    print(f"Metrics       : {metrics_path}")
    print(f"Confusion     : {confusion_path}")
    print(f"Summary       : {summary_path}")
    print("=" * 96)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
