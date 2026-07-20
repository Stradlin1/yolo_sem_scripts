#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 Bucket 二分类语义分割模型推理图片，并将 Bucket 结果合并到原六类标签。

原始六类标签协议：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background

Bucket 模型内部协议：
    0 = NotBucket
    1 = Bucket

最终七类标签协议：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background
    6 = Bucket

默认输入：
    best.pt
    bev_sem_round2_merged/images/train/2026-06-28_01-08-45
    bev_sem_round2_merged/masks/train/2026-06-28_01-08-45

默认输出：
    auto7/masks7/train/2026-06-28_01-08-45
    auto7/overlays7/train/2026-06-28_01-08-45

默认路径均相对于本脚本所在目录解析。

修正版：兼容 Ultralytics 8.4.x 中 Masks/BaseTensor 包装对象。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np


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


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "使用 Bucket 二分类模型推理，并把 Bucket 区域覆盖到原六类 mask 中。"
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=script_dir / "best.pt",
        help="Bucket 二分类模型。默认：脚本目录/best.pt",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=(
            script_dir
            / "bev_sem_round2_merged"
            / "images"
            / "train"
            / "2026-06-28_01-08-45"
        ),
        help="输入图片目录。",
    )
    parser.add_argument(
        "--base-masks",
        type=Path,
        default=(
            script_dir
            / "bev_sem_round2_merged"
            / "masks"
            / "train"
            / "2026-06-28_01-08-45"
        ),
        help="原始六类标签目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_dir / "auto7",
        help="输出根目录。默认：脚本目录/auto7",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="输出目录中的 split 名称，默认 train。",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="2026-06-28_01-08-45",
        help="输出目录中的 run 名称。",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="模型推理尺寸，默认 320。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="推理设备，例如 0 或 cpu。默认 0。",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help=(
            "实例分割兼容分支的置信度阈值。"
            "纯语义输出通常不使用该值，默认 0.25。"
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.48,
        help="Overlay 颜色透明度，默认 0.48。",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=0,
        help=(
            "删除面积小于该值的 Bucket 连通区域，单位为像素。"
            "默认 0，表示不删除。"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的 masks7 和 overlays7。",
    )
    return parser.parse_args()


def resolve_path(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def iter_images(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def to_numpy(value: Any) -> np.ndarray:
    """
    将 torch.Tensor、numpy.ndarray 和 Ultralytics BaseTensor 包装对象
    转换为 numpy.ndarray。

    注意：必须先识别 torch.Tensor，再访问通用对象的 .data。
    torch.Tensor 本身也有 .data 属性；如果先递归访问 tensor.data，
    会不断生成新的 Tensor 包装并造成递归失败。
    """
    if isinstance(value, np.ndarray):
        return value

    if value is None:
        raise TypeError("不能把 None 转换为 numpy.ndarray")

    # 必须优先处理原生 torch.Tensor。
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass

    # 兼容列表或元组中包裹单个 Tensor/数组的情况。
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return to_numpy(value[0])

        arrays = [to_numpy(item) for item in value]
        try:
            return np.stack(arrays, axis=0)
        except ValueError as error:
            raise TypeError(
                "列表或元组中的模型输出形状不一致，无法堆叠："
                f"type={type(value).__name__}, length={len(value)}"
            ) from error

    # 兼容字典包装。优先检查常见的语义输出键。
    if isinstance(value, dict):
        preferred_keys = (
            "semantic_mask",
            "mask",
            "masks",
            "pred",
            "prediction",
            "output",
            "data",
            "logits",
            "probs",
        )
        for key in preferred_keys:
            if key in value and value[key] is not None:
                return to_numpy(value[key])

        if len(value) == 1:
            return to_numpy(next(iter(value.values())))

        raise TypeError(
            "模型输出是字典，但未找到可识别的语义 mask 字段："
            f"keys={sorted(value.keys())}"
        )

    # Ultralytics SemanticMask/BaseTensor 的真实 Tensor 位于 .data。
    data = getattr(value, "data", None)
    if data is not None and data is not value:
        return to_numpy(data)

    # 兼容其他具有 detach/cpu/numpy 接口的张量类对象。
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        converted = value.numpy()
        if isinstance(converted, np.ndarray):
            return converted
        if converted is not value:
            return to_numpy(converted)

    try:
        array = np.asarray(value)
    except Exception as error:
        raise TypeError(
            "无法转换模型输出为 numpy.ndarray："
            f"type={type(value).__name__}"
        ) from error

    if array.dtype == object:
        raise TypeError(
            "模型输出被转换成 object 数组，说明仍有包装层未解开："
            f"type={type(value).__name__}"
        )

    return array



def resize_nearest(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
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
    将常见语义输出转换为 Bucket 二值 mask：
    - [H, W]：类别 ID 图或概率图；
    - [1, H, W]：单通道类别图/概率图；
    - [2, H, W]：两类 logits/probabilities；
    - [H, W, 2]：两类 logits/probabilities；
    - [1, 2, H, W]：带 batch 的两类输出。
    """
    array = np.asarray(array)

    while array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 2:
        array = resize_nearest(array, target_hw)

        if np.issubdtype(array.dtype, np.floating):
            finite = array[np.isfinite(array)]
            if finite.size == 0:
                raise RuntimeError("模型输出全是 NaN 或 Inf。")

            min_value = float(finite.min())
            max_value = float(finite.max())

            # 0～1 的浮点单通道输出按 Bucket 概率处理。
            if 0.0 <= min_value and max_value <= 1.0:
                return (array >= 0.5).astype(np.uint8)

        # 整数类别图，或浮点类别 ID 图。
        rounded = np.rint(array).astype(np.int32)
        unique_ids = set(np.unique(rounded).tolist())

        if unique_ids.issubset({0, 1}):
            return (rounded == BUCKET_MODEL_ID).astype(np.uint8)

        raise RuntimeError(
            "二维模型输出无法解释为 Bucket 二分类结果，"
            f"shape={array.shape}, dtype={array.dtype}, "
            f"unique={sorted(unique_ids)[:20]}"
        )

    if array.ndim == 3:
        # CHW：2 类 logits/probabilities。
        if array.shape[0] == 2:
            class_map = np.argmax(array, axis=0).astype(np.uint8)
            class_map = resize_nearest(class_map, target_hw)
            return (class_map == BUCKET_MODEL_ID).astype(np.uint8)

        # HWC：2 类 logits/probabilities。
        if array.shape[-1] == 2:
            class_map = np.argmax(array, axis=-1).astype(np.uint8)
            class_map = resize_nearest(class_map, target_hw)
            return (class_map == BUCKET_MODEL_ID).astype(np.uint8)

        if array.shape[0] == 1:
            return decode_array_prediction(array[0], target_hw)

    if array.ndim == 4 and array.shape[0] == 1:
        return decode_array_prediction(array[0], target_hw)

    raise RuntimeError(
        "无法解析模型数组输出："
        f"shape={array.shape}, dtype={array.dtype}"
    )


def decode_ultralytics_result(
    result: Any,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """
    兼容语义分割 Results 和标准实例分割 Results。

    优先级：
    1. sem_seg / semantic / pred_semantic 等语义输出；
    2. masks.data 为单张语义类别图；
    3. 标准实例分割：合并 boxes.cls == 1 的实例 mask。
    """
    semantic_attributes = (
        "sem_seg",
        "semantic",
        "semantic_mask",
        "pred_semantic",
        "segmentation",
    )

    for name in semantic_attributes:
        value = getattr(result, name, None)
        if value is not None:
            try:
                return decode_array_prediction(to_numpy(value), target_hw)
            except Exception:
                # 某些 Ultralytics 版本中的 semantic 属性是包装对象，
                # 若当前字段无法解释，则继续尝试 masks/pred 等字段。
                pass

    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)

    if masks_obj is not None:
        data = getattr(masks_obj, "data", masks_obj)
        try:
            masks_array = to_numpy(data)
        except Exception as error:
            raise RuntimeError(
                "检测到 result.masks，但无法解包其底层数据："
                f"masks_type={type(masks_obj).__name__}, "
                f"data_type={type(data).__name__}"
            ) from error

        # 语义模型常见输出：[1, H, W]，内容是 0/1 类别 ID。
        if masks_array.ndim == 3 and masks_array.shape[0] == 1:
            return decode_array_prediction(masks_array[0], target_hw)

        # 标准实例分割兼容：将类别 1 的实例合并为 Bucket。
        if (
            masks_array.ndim == 3
            and boxes_obj is not None
            and getattr(boxes_obj, "cls", None) is not None
        ):
            classes = to_numpy(boxes_obj.cls).reshape(-1).astype(np.int32)

            if len(classes) != masks_array.shape[0]:
                raise RuntimeError(
                    "实例 mask 数量与 boxes.cls 数量不一致："
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

    # 少数自定义 predictor 可能直接把预测挂在 result.pred。
    pred = getattr(result, "pred", None)
    if pred is not None:
        return decode_array_prediction(to_numpy(pred), target_hw)

    available = sorted(
        key for key in getattr(result, "__dict__", {}).keys()
        if not key.startswith("_")
    )
    raise RuntimeError(
        "无法从 Ultralytics Results 中提取语义 mask。\n"
        f"可用字段：{available}\n"
        "请打印 result 后，根据实际输出字段调整 decode_ultralytics_result()。"
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

    # 自定义语义模型可能直接返回 ndarray/tensor。
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


def remove_small_components(
    binary_mask: np.ndarray,
    min_area: int,
) -> np.ndarray:
    if min_area <= 0:
        return binary_mask

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


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def read_old_mask(path: Path, expected_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        raise RuntimeError(f"无法读取原标签：{path}")

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
            f"图片和原标签尺寸不一致：{path}，"
            f"mask={mask.shape}, image={expected_hw}"
        )

    ids = set(np.unique(mask).tolist())
    illegal = ids - VALID_OLD_IDS
    if illegal:
        raise RuntimeError(
            f"原六类标签包含非法 ID：{path}，"
            f"全部 ID={sorted(ids)}，非法 ID={sorted(illegal)}"
        )

    return mask


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, bgr in CLASS_COLORS_BGR.items():
        color[mask == class_id] = bgr

    return color


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

    bucket_binary = (final_mask == BUCKET_GLOBAL_ID).astype(np.uint8)
    contours, _ = cv2.findContours(
        bucket_binary,
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

    # 在左侧绘制类别图例。
    line_height = 18
    legend_width = 178
    legend_height = 7 * line_height + 8

    panel = overlay.copy()
    cv2.rectangle(
        panel,
        (0, 0),
        (legend_width, legend_height),
        (0, 0, 0),
        thickness=-1,
    )
    overlay = cv2.addWeighted(overlay, 0.35, panel, 0.65, 0.0)

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
        return {index: str(value) for index, value in enumerate(raw_names)}
    return {}


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    model_path = resolve_path(args.model, script_dir)
    image_root = resolve_path(args.images, script_dir)
    base_mask_root = resolve_path(args.base_masks, script_dir)
    output_root = resolve_path(args.output_root, script_dir)

    output_mask_root = (
        output_root / "masks7" / args.split / args.run_name
    )
    output_overlay_root = (
        output_root / "overlays7" / args.split / args.run_name
    )
    manifest_path = output_root / f"manifest_{args.split}_{args.run_name}.csv"

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError(f"--alpha 必须在 0～1，当前为 {args.alpha}")

    if args.min_area < 0:
        raise ValueError(f"--min-area 不能为负数，当前为 {args.min_area}")

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{image_root}")
    if not base_mask_root.is_dir():
        raise FileNotFoundError(f"原标签目录不存在：{base_mask_root}")

    image_paths = iter_images(image_root)
    if not image_paths:
        raise RuntimeError(f"没有找到图片：{image_root}")

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "当前 Python 环境没有安装 ultralytics。"
            "请在训练 Bucket 模型时使用的环境中运行。"
        ) from error

    print("=" * 80)
    print("Bucket 自动标签融合")
    print(f"Ultralytics  : {ultralytics.__version__}")
    print(f"模型          : {model_path}")
    print(f"图片          : {image_root}")
    print(f"原六类标签    : {base_mask_root}")
    print(f"输出 masks7   : {output_mask_root}")
    print(f"输出 overlays7: {output_overlay_root}")
    print(f"图片数量      : {len(image_paths)}")
    print(f"imgsz         : {args.imgsz}")
    print(f"device        : {args.device}")
    print(f"min_area      : {args.min_area}")
    print("=" * 80)

    model = YOLO(str(model_path))
    model_names = normalize_model_names(getattr(model, "names", {}))
    print(f"模型类别      : {model_names}")

    if model_names and set(model_names) != {0, 1}:
        raise RuntimeError(
            "当前 best.pt 不是二分类 Bucket 模型。\n"
            f"模型类别：{model_names}\n"
            "期望：0=NotBucket，1=Bucket。"
        )

    records: list[dict[str, Any]] = []
    missing_masks: list[Path] = []

    for index, image_path in enumerate(image_paths, start=1):
        relative_image = image_path.relative_to(image_root)
        relative_mask = relative_image.with_suffix(".png")

        base_mask_path = base_mask_root / relative_mask
        output_mask_path = output_mask_root / relative_mask
        output_overlay_path = output_overlay_root / relative_mask

        if not base_mask_path.is_file():
            missing_masks.append(base_mask_path)
            continue

        if (
            output_mask_path.exists()
            and output_overlay_path.exists()
            and not args.overwrite
        ):
            print(
                f"[跳过] {index}/{len(image_paths)} "
                f"{relative_image.as_posix()}，输出已存在"
            )
            continue

        image = read_image(image_path)
        target_hw = image.shape[:2]
        old_mask = read_old_mask(base_mask_path, target_hw)

        bucket_mask = predict_bucket_mask(
            model=model,
            image_path=image_path,
            target_hw=target_hw,
            imgsz=args.imgsz,
            device=args.device,
            conf=args.conf,
        )
        bucket_mask = remove_small_components(
            bucket_mask,
            args.min_area,
        )

        bucket_ids = set(np.unique(bucket_mask).tolist())
        if not bucket_ids.issubset({0, 1}):
            raise RuntimeError(
                f"Bucket 预测不是二值 mask：{image_path}，"
                f"IDs={sorted(bucket_ids)}"
            )

        final_mask = old_mask.copy()
        final_mask[bucket_mask == BUCKET_MODEL_ID] = BUCKET_GLOBAL_ID

        final_ids = set(np.unique(final_mask).tolist())
        illegal_final = final_ids - VALID_FINAL_IDS
        if illegal_final:
            raise RuntimeError(
                f"融合结果出现非法 ID：{image_path}，"
                f"IDs={sorted(final_ids)}"
            )

        overlay = make_overlay(image, final_mask, args.alpha)

        write_png(output_mask_path, final_mask)
        write_png(output_overlay_path, overlay)

        bucket_pixels = int(np.count_nonzero(bucket_mask == 1))
        records.append(
            {
                "image": relative_image.as_posix(),
                "base_mask": relative_mask.as_posix(),
                "output_mask7": relative_mask.as_posix(),
                "bucket_pixels": bucket_pixels,
                "has_bucket": int(bucket_pixels > 0),
            }
        )

        print(
            f"[完成] {index}/{len(image_paths)} "
            f"{relative_image.as_posix()} "
            f"Bucket pixels={bucket_pixels}"
        )

    if missing_masks:
        preview = "\n".join(str(path) for path in missing_masks[:20])
        suffix = ""
        if len(missing_masks) > 20:
            suffix = f"\n……另外还有 {len(missing_masks) - 20} 个缺失标签。"
        raise RuntimeError(
            f"有 {len(missing_masks)} 张图片缺少原六类标签：\n"
            f"{preview}{suffix}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "base_mask",
                "output_mask7",
                "bucket_pixels",
                "has_bucket",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    positive_images = sum(int(row["has_bucket"]) for row in records)
    bucket_pixels = sum(int(row["bucket_pixels"]) for row in records)

    print("=" * 80)
    print("推理与融合完成")
    print(f"本次生成数量      : {len(records)}")
    print(f"预测含 Bucket 图片: {positive_images}")
    print(f"Bucket 像素总数   : {bucket_pixels}")
    print(f"输出 masks7       : {output_mask_root}")
    print(f"输出 overlays7    : {output_overlay_root}")
    print(f"manifest          : {manifest_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
