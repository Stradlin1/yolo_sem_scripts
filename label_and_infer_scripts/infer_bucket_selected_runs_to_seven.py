#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Bucket 二分类语义分割模型，为指定 train/val run 生成完整七类标签。

输入根目录：
    /home/xhm/Desktop/bucket_label/bev_sem_round2_merged

输入结构：
    images/train/<run_name>/
    images/val/<run_name>/
    masks/train/<run_name>/
    masks/val/<run_name>/

处理的 run：
    2026-06-28_01-05-54
    2026-06-28_01-08-45
    2026-06-28_01-13-14
    2026-06-28_01-18-35
    2026-06-28_01-25-30

输出结构：
    /home/xhm/Desktop/bucket_label/label_seven/
    ├── classseventh/
    │   ├── train/<run_name>/
    │   └── val/<run_name>/
    ├── overlays/
    │   ├── train/<run_name>/
    │   └── val/<run_name>/
    └── manifest.csv

类别协议：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background
    6 = Bucket

融合规则：
    final_mask = old_mask.copy()
    final_mask[bucket_prediction == 1] = 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# =============================================================================
# 固定配置
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = SCRIPT_DIR / "best.pt"
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "bev_sem_round2_merged"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "label_seven"

RUN_NAMES = [
    "2026-06-28_01-05-54",
    "2026-06-28_01-08-45",
    "2026-06-28_01-13-14",
    "2026-06-28_01-18-35",
    "2026-06-28_01-25-30",
]

SPLITS = ["train", "val"]

CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
    6: "Bucket",
}

# OpenCV 使用 BGR。
CLASS_COLORS_BGR = {
    0: (180, 70, 220),    # JinQu
    1: (255, 180, 40),    # C_ZhenLiaoShi
    2: (60, 210, 90),     # B_TongDao
    3: (40, 170, 255),    # A_DaTing
    4: (255, 90, 70),     # P
    5: (45, 45, 45),      # background
    6: (0, 0, 255),       # Bucket
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_OLD_IDS = set(range(6))
VALID_FINAL_IDS = set(range(7))

BUCKET_MODEL_ID = 1
BUCKET_GLOBAL_ID = 6


# =============================================================================
# 参数
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用 Bucket 二分类模型处理指定 train/val run，"
            "并与原六类 mask 融合成完整七类标签。"
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Bucket 二分类 best.pt，默认：脚本目录/best.pt",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="bev_sem_round2_merged 根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录，默认：脚本目录/label_seven",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="推理尺寸，默认 320。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，例如 0 或 cpu，默认 0。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="兼容实例分割输出时使用的置信度，默认 0.25。",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Overlay 透明度，默认 0.48。",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help="删除小于该像素面积的 Bucket 连通域，默认 0 表示不删除。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出。",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "若任意 run 在 train 或 val 中不存在，则立即报错。"
            "默认只打印跳过信息并继续。"
        ),
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


# =============================================================================
# 模型输出解码
# =============================================================================

def to_numpy(value: Any) -> np.ndarray:
    """
    兼容 numpy、torch.Tensor、Ultralytics BaseTensor/SemanticMask。
    """
    if isinstance(value, np.ndarray):
        return value

    if value is None:
        raise TypeError("不能把 None 转换成 numpy.ndarray")

    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return to_numpy(value[0])

        arrays = [to_numpy(item) for item in value]
        return np.stack(arrays, axis=0)

    if isinstance(value, dict):
        for key in (
            "semantic_mask",
            "mask",
            "masks",
            "pred",
            "prediction",
            "output",
            "data",
            "logits",
            "probs",
        ):
            if key in value and value[key] is not None:
                return to_numpy(value[key])

        if len(value) == 1:
            return to_numpy(next(iter(value.values())))

        raise TypeError(
            f"无法识别字典模型输出，keys={sorted(value.keys())}"
        )

    data = getattr(value, "data", None)
    if data is not None and data is not value:
        return to_numpy(data)

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        converted = value.numpy()
        if isinstance(converted, np.ndarray):
            return converted
        return to_numpy(converted)

    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(
            f"模型输出仍是 object 数组，原类型={type(value).__name__}"
        )
    return array


def resize_nearest(
    mask: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    target_h, target_w = target_hw
    if mask.shape == (target_h, target_w):
        return mask

    return cv2.resize(
        mask,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )


def decode_array_prediction(
    array: np.ndarray,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    输出统一转换为：
        0 = NotBucket
        1 = Bucket
    """
    array = np.asarray(array)

    while array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 2:
        array = resize_nearest(array, target_hw)

        if np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise RuntimeError("模型输出全部为 NaN 或 Inf。")

            min_value = float(finite.min())
            max_value = float(finite.max())

            if 0.0 <= min_value and max_value <= 1.0:
                return (array >= 0.5).astype(np.uint8)

        rounded = np.rint(array).astype(np.int32)
        unique_ids = set(np.unique(rounded).tolist())

        if unique_ids.issubset({0, 1}):
            return (rounded == BUCKET_MODEL_ID).astype(np.uint8)

        raise RuntimeError(
            "二维模型输出不是二分类类别图："
            f"shape={array.shape}, dtype={array.dtype}, "
            f"unique={sorted(unique_ids)[:20]}"
        )

    if array.ndim == 3:
        # CHW，两类 logits/probabilities。
        if array.shape[0] == 2:
            class_map = np.argmax(array, axis=0).astype(np.uint8)
            class_map = resize_nearest(class_map, target_hw)
            return (class_map == BUCKET_MODEL_ID).astype(np.uint8)

        # HWC，两类 logits/probabilities。
        if array.shape[-1] == 2:
            class_map = np.argmax(array, axis=-1).astype(np.uint8)
            class_map = resize_nearest(class_map, target_hw)
            return (class_map == BUCKET_MODEL_ID).astype(np.uint8)

        if array.shape[0] == 1:
            return decode_array_prediction(array[0], target_hw)

    if array.ndim == 4 and array.shape[0] == 1:
        return decode_array_prediction(array[0], target_hw)

    raise RuntimeError(
        "无法解析模型输出数组："
        f"shape={array.shape}, dtype={array.dtype}"
    )


def decode_ultralytics_result(
    result: Any,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    优先读取当前 yolo_sem 使用的 result.semantic_mask。
    同时保留对其他常见输出字段的兼容。
    """
    for name in (
        "semantic_mask",
        "sem_seg",
        "semantic",
        "pred_semantic",
        "segmentation",
    ):
        value = getattr(result, name, None)
        if value is not None:
            try:
                return decode_array_prediction(to_numpy(value), target_hw)
            except (TypeError, RuntimeError, ValueError):
                pass

    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)

    if masks_obj is not None:
        masks_array = to_numpy(masks_obj)

        if masks_array.ndim == 3 and masks_array.shape[0] == 1:
            return decode_array_prediction(masks_array[0], target_hw)

        # 实例分割兼容分支。
        if (
            masks_array.ndim == 3
            and boxes_obj is not None
            and getattr(boxes_obj, "cls", None) is not None
        ):
            classes = to_numpy(boxes_obj.cls).reshape(-1).astype(np.int32)

            if len(classes) != masks_array.shape[0]:
                raise RuntimeError(
                    "实例 mask 数量和类别数量不一致："
                    f"masks={masks_array.shape[0]}, classes={len(classes)}"
                )

            bucket = np.zeros(masks_array.shape[1:], dtype=np.uint8)
            for instance_mask, class_id in zip(masks_array, classes):
                if int(class_id) == BUCKET_MODEL_ID:
                    bucket[instance_mask > 0.5] = 1

            return resize_nearest(bucket, target_hw).astype(np.uint8)

        try:
            return decode_array_prediction(masks_array, target_hw)
        except RuntimeError:
            pass

    pred = getattr(result, "pred", None)
    if pred is not None:
        return decode_array_prediction(to_numpy(pred), target_hw)

    available = sorted(
        key
        for key in getattr(result, "__dict__", {}).keys()
        if not key.startswith("_")
    )
    raise RuntimeError(
        "无法从 Ultralytics Results 中提取语义 mask。\n"
        f"可用字段：{available}"
    )


def predict_bucket_mask(
    model: Any,
    image_path: Path,
    target_hw: tuple[int, int],
    imgsz: int,
    device: str,
    conf: float,
) -> np.ndarray:
    predictions = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        device=device,
        conf=conf,
        save=False,
        verbose=False,
    )

    if predictions is None:
        raise RuntimeError(f"模型没有返回结果：{image_path}")

    if isinstance(predictions, np.ndarray) or hasattr(predictions, "detach"):
        return decode_array_prediction(to_numpy(predictions), target_hw)

    if not isinstance(predictions, (list, tuple)):
        try:
            return decode_array_prediction(to_numpy(predictions), target_hw)
        except Exception:
            pass

    if len(predictions) == 0:
        raise RuntimeError(f"模型返回空结果：{image_path}")

    first = predictions[0]

    if isinstance(first, np.ndarray) or hasattr(first, "detach"):
        return decode_array_prediction(to_numpy(first), target_hw)

    return decode_ultralytics_result(first, target_hw)


# =============================================================================
# 图像、mask 与 overlay
# =============================================================================

def iter_images(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def read_old_mask(
    path: Path,
    expected_hw: tuple[int, int],
) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f"无法读取原六类标签：{path}")

    if mask.ndim != 2:
        raise RuntimeError(
            f"原标签必须是单通道：{path}，shape={mask.shape}"
        )

    if mask.dtype != np.uint8:
        raise RuntimeError(
            f"原标签必须为 uint8：{path}，dtype={mask.dtype}"
        )

    if mask.shape != expected_hw:
        raise RuntimeError(
            f"图片和标签尺寸不一致：{path}，"
            f"mask={mask.shape}, image={expected_hw}"
        )

    ids = set(np.unique(mask).tolist())
    illegal_ids = ids - VALID_OLD_IDS
    if illegal_ids:
        raise RuntimeError(
            f"原标签必须只包含 0～5：{path}，"
            f"全部 ID={sorted(ids)}，非法 ID={sorted(illegal_ids)}"
        )

    return mask


def remove_small_components(
    binary_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    if min_area <= 0:
        return binary_mask.astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8),
        connectivity=8,
    )

    cleaned = np.zeros_like(binary_mask, dtype=np.uint8)

    for label_id in range(1, count):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label_id] = 1

    return cleaned


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, color in CLASS_COLORS_BGR.items():
        color_mask[mask == class_id] = color

    return color_mask


def make_overlay(
    image: np.ndarray,
    final_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    color_mask = colorize_mask(final_mask)

    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        color_mask,
        alpha,
        0.0,
    )

    # Bucket 边界额外加粗。
    bucket_mask = (final_mask == BUCKET_GLOBAL_ID).astype(np.uint8)
    contours, _ = cv2.findContours(
        bucket_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(
        overlay,
        contours,
        -1,
        CLASS_COLORS_BGR[BUCKET_GLOBAL_ID],
        2,
        lineType=cv2.LINE_AA,
    )

    # 左上角类别图例。
    line_height = 18
    panel_width = 180
    panel_height = 7 * line_height + 8

    dark_panel = overlay.copy()
    cv2.rectangle(
        dark_panel,
        (0, 0),
        (panel_width, panel_height),
        (0, 0, 0),
        thickness=-1,
    )
    overlay = cv2.addWeighted(
        overlay,
        0.35,
        dark_panel,
        0.65,
        0.0,
    )

    for row, class_id in enumerate(range(7)):
        y = 15 + row * line_height
        color = CLASS_COLORS_BGR[class_id]

        cv2.rectangle(
            overlay,
            (6, y - 10),
            (19, y + 3),
            color,
            thickness=-1,
        )
        cv2.putText(
            overlay,
            f"{class_id}: {CLASS_NAMES[class_id]}",
            (25, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return overlay


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"保存失败：{path}")


def normalize_model_names(raw_names: Any) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}

    if isinstance(raw_names, list):
        return {
            index: str(value)
            for index, value in enumerate(raw_names)
        }

    return {}


# =============================================================================
# 单个 run 处理
# =============================================================================

def process_run(
    *,
    model: Any,
    split: str,
    run_name: str,
    dataset_root: Path,
    output_root: Path,
    imgsz: int,
    device: str,
    conf: float,
    alpha: float,
    min_area: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    image_root = dataset_root / "images" / split / run_name
    mask_root = dataset_root / "masks" / split / run_name

    output_mask_root = (
        output_root / "classseventh" / split / run_name
    )
    output_overlay_root = (
        output_root / "overlays" / split / run_name
    )

    if not image_root.is_dir() or not mask_root.is_dir():
        return []

    image_paths = iter_images(image_root)
    if not image_paths:
        print(f"[跳过] {split}/{run_name} 中没有图片。")
        return []

    print("-" * 80)
    print(f"开始处理：{split}/{run_name}")
    print(f"图片目录：{image_root}")
    print(f"标签目录：{mask_root}")
    print(f"图片数量：{len(image_paths)}")

    records: list[dict[str, Any]] = []
    missing_masks: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        relative_image = image_path.relative_to(image_root)
        relative_mask = relative_image.with_suffix(".png")

        old_mask_path = mask_root / relative_mask
        output_mask_path = output_mask_root / relative_mask
        output_overlay_path = output_overlay_root / relative_mask

        if not old_mask_path.is_file():
            missing_masks.append(old_mask_path)
            continue

        if (
            output_mask_path.exists()
            and output_overlay_path.exists()
            and not overwrite
        ):
            print(
                f"[跳过已有] {split}/{run_name} "
                f"{index}/{len(image_paths)} "
                f"{relative_image.as_posix()}"
            )
            continue

        image = read_image(image_path)
        old_mask = read_old_mask(
            old_mask_path,
            image.shape[:2],
        )

        bucket_mask = predict_bucket_mask(
            model=model,
            image_path=image_path,
            target_hw=image.shape[:2],
            imgsz=imgsz,
            device=device,
            conf=conf,
        )
        bucket_mask = remove_small_components(
            bucket_mask,
            min_area,
        )

        bucket_ids = set(np.unique(bucket_mask).tolist())
        if not bucket_ids.issubset({0, 1}):
            raise RuntimeError(
                f"Bucket 预测不是二值标签：{image_path}，"
                f"IDs={sorted(bucket_ids)}"
            )

        final_mask = old_mask.copy()
        final_mask[bucket_mask == BUCKET_MODEL_ID] = BUCKET_GLOBAL_ID

        final_ids = set(np.unique(final_mask).tolist())
        illegal_final_ids = final_ids - VALID_FINAL_IDS
        if illegal_final_ids:
            raise RuntimeError(
                f"融合结果包含非法类别：{image_path}，"
                f"IDs={sorted(final_ids)}"
            )

        overlay = make_overlay(
            image,
            final_mask,
            alpha,
        )

        write_png(output_mask_path, final_mask)
        write_png(output_overlay_path, overlay)

        bucket_pixels = int(np.count_nonzero(bucket_mask == 1))

        records.append(
            {
                "split": split,
                "run": run_name,
                "image": relative_image.as_posix(),
                "source_mask": relative_mask.as_posix(),
                "output_mask": relative_mask.as_posix(),
                "has_bucket": int(bucket_pixels > 0),
                "bucket_pixels": bucket_pixels,
            }
        )

        print(
            f"[完成] {split}/{run_name} "
            f"{index}/{len(image_paths)} "
            f"{relative_image.as_posix()} "
            f"Bucket pixels={bucket_pixels}"
        )

    if missing_masks:
        preview = "\n".join(str(path) for path in missing_masks[:20])
        suffix = ""
        if len(missing_masks) > 20:
            suffix = (
                f"\n……另外还有 {len(missing_masks) - 20} 个缺失标签。"
            )
        raise RuntimeError(
            f"{split}/{run_name} 有 {len(missing_masks)} 张图片缺少标签：\n"
            f"{preview}{suffix}"
        )

    return records


# =============================================================================
# 主程序
# =============================================================================

def main() -> None:
    args = parse_args()

    model_path = resolve_path(args.model, SCRIPT_DIR)
    dataset_root = resolve_path(args.dataset_root, SCRIPT_DIR)
    output_root = resolve_path(args.output_root, SCRIPT_DIR)

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据集根目录不存在：{dataset_root}")

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha 必须在 0～1，当前为 {args.alpha}")

    if args.min_area < 0:
        raise ValueError(
            f"--min-area 不能为负数，当前为 {args.min_area}"
        )

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "当前 Python 环境没有安装 ultralytics。"
            "请在 yolo_sem 环境中运行。"
        ) from error

    print("=" * 80)
    print("指定 run 的 Bucket 推理与七类标签融合")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"模型         : {model_path}")
    print(f"数据集       : {dataset_root}")
    print(f"输出标签     : {output_root / 'classseventh'}")
    print(f"输出预览     : {output_root / 'overlays'}")
    print(f"处理 split   : {SPLITS}")
    print(f"处理 run     : {RUN_NAMES}")
    print(f"imgsz        : {args.imgsz}")
    print(f"device       : {args.device}")
    print(f"min_area     : {args.min_area}")
    print("=" * 80)

    model = YOLO(str(model_path))

    model_names = normalize_model_names(
        getattr(model, "names", {})
    )
    print(f"模型类别     : {model_names}")

    if model_names and set(model_names) != {0, 1}:
        raise RuntimeError(
            "当前模型不是 Bucket 二分类模型。\n"
            f"实际类别：{model_names}\n"
            "期望类别：0=NotBucket，1=Bucket。"
        )

    all_records: list[dict[str, Any]] = []
    missing_pairs: list[str] = []

    for split in SPLITS:
        for run_name in RUN_NAMES:
            image_root = dataset_root / "images" / split / run_name
            mask_root = dataset_root / "masks" / split / run_name

            if not image_root.is_dir() or not mask_root.is_dir():
                message = (
                    f"{split}/{run_name} 不完整："
                    f"images_exists={image_root.is_dir()}, "
                    f"masks_exists={mask_root.is_dir()}"
                )
                missing_pairs.append(message)

                if args.strict:
                    raise FileNotFoundError(message)

                print(f"[跳过不存在] {message}")
                continue

            records = process_run(
                model=model,
                split=split,
                run_name=run_name,
                dataset_root=dataset_root,
                output_root=output_root,
                imgsz=args.imgsz,
                device=args.device,
                conf=args.conf,
                alpha=args.alpha,
                min_area=args.min_area,
                overwrite=args.overwrite,
            )
            all_records.extend(records)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"

    fieldnames = [
        "split",
        "run",
        "image",
        "source_mask",
        "output_mask",
        "has_bucket",
        "bucket_pixels",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(all_records)

    total_images = len(all_records)
    positive_images = sum(
        int(record["has_bucket"])
        for record in all_records
    )
    total_bucket_pixels = sum(
        int(record["bucket_pixels"])
        for record in all_records
    )

    print("=" * 80)
    print("全部处理完成")
    print(f"本次新生成标签数  : {total_images}")
    print(f"预测含 Bucket 图片: {positive_images}")
    print(f"Bucket 像素总数   : {total_bucket_pixels}")
    print(f"七类标签目录      : {output_root / 'classseventh'}")
    print(f"Overlay 目录      : {output_root / 'overlays'}")
    print(f"Manifest          : {manifest_path}")

    if missing_pairs:
        print("-" * 80)
        print("以下 split/run 不存在或图片标签目录不完整，已跳过：")
        for message in missing_pairs:
            print(f"  {message}")

    print("=" * 80)


if __name__ == "__main__":
    main()
