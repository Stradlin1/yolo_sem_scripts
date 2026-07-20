#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
正式训练脚本：在合并后的第二轮 BEV 语义分割数据集上继续训练。

训练起点：
    /home/xhm/Desktop/relocate_ws/models/071801.pt

数据集：
    /home/xhm/Desktop/sem_train/datasets/bev_sem_round2_merged/data.yaml

输出：
    /home/xhm/Desktop/sem_train/runs/round2_merged_071801_finetune/

最佳权重额外复制到：
    /home/xhm/Desktop/sem_train/models/0718_round2_best.pt

标签格式：
    单通道 uint8 PNG，每个像素是类别 ID：
      0 background
      1 P
      2 A_DaTing
      3 B_TongDao
      4 C_ZhenLiaoShi
      5 JinQu
"""

from __future__ import annotations

import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml
from ultralytics import YOLO


# =============================================================================
# 固定路径
# =============================================================================

WORKSPACE = Path("/home/xhm/Desktop/sem_train")

# 第二轮训练从上一轮训练出的权重继续微调，不重新从官方基础权重开始。
MODEL_PATH = Path("/home/xhm/Desktop/relocate_ws/models/071801.pt")

DATASET_ROOT = WORKSPACE / "datasets" / "bev_sem_round2_merged"
DATA_YAML = DATASET_ROOT / "data.yaml"

PROJECT_DIR = WORKSPACE / "runs"
RUN_NAME = "round2_merged_071801_finetune"
RUN_DIR = PROJECT_DIR / RUN_NAME

# 训练结束后复制一份便于下一轮使用；不会覆盖上一轮 071801.pt。
EXPORTED_BEST = WORKSPACE / "models" / "0718_round2_best.pt"


# =============================================================================
# 类别定义
# =============================================================================

EXPECTED_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

VALID_CLASS_IDS = set(EXPECTED_NAMES)
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


# =============================================================================
# 训练参数
# =============================================================================

IMAGE_SIZE = 320
EPOCHS = 100
BATCH_SIZE = 8
DEVICE = 0
WORKERS = 8
SEED = 42

TRAIN_ARGS = {
    "data": str(DATA_YAML),
    "imgsz": IMAGE_SIZE,
    "epochs": EPOCHS,
    "batch": BATCH_SIZE,
    "device": DEVICE,
    "workers": WORKERS,

    "project": str(PROJECT_DIR),
    "name": RUN_NAME,
    "exist_ok": False,

    # 从 071801.pt 继续微调。
    "pretrained": True,
    "resume": False,

    # 稳定训练设置。
    "optimizer": "auto",
    "amp": True,
    "seed": SEED,
    "deterministic": True,
    "patience": 25,

    # 输出设置。
    "save": True,
    "save_period": 10,
    "plots": True,
    "verbose": True,
    "cache": False,

    # BEV 方位和地面纹理具有物理含义，禁止翻转、旋转、透视和拼接增强。
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}


def normalize_names(raw_names: object) -> dict[int, str]:
    """把模型或 YAML 中的 names 统一为 {int: str}。"""
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}

    if isinstance(raw_names, (list, tuple)):
        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(raw_names)
        }

    raise TypeError(f"不支持的 names 类型: {type(raw_names).__name__}")


def list_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def mask_path_for_image(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
) -> Path:
    relative = image_path.relative_to(image_root)
    return (mask_root / relative).with_suffix(".png")


def load_dataset_yaml() -> dict:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"找不到数据集 YAML: {DATA_YAML}")

    with DATA_YAML.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise RuntimeError(f"data.yaml 内容无效: {DATA_YAML}")

    yaml_names = normalize_names(data.get("names"))
    if yaml_names != EXPECTED_NAMES:
        raise RuntimeError(
            "data.yaml 的类别顺序不正确。\n"
            f"实际: {yaml_names}\n"
            f"要求: {EXPECTED_NAMES}"
        )

    return data


def validate_split(split: str) -> tuple[int, Counter[int]]:
    """
    验证一个 split：
      - 图片和 mask 一一对应；
      - mask 为单通道；
      - 图片与 mask 尺寸一致；
      - mask 像素值只能是 0..5。
    """
    image_root = DATASET_ROOT / "images" / split
    mask_root = DATASET_ROOT / "masks" / split

    if not image_root.is_dir():
        raise FileNotFoundError(f"找不到图片目录: {image_root}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"找不到标签目录: {mask_root}")

    images = list_images(image_root)
    if not images:
        raise RuntimeError(f"{split} 中没有图片: {image_root}")

    expected_masks = {
        mask_path_for_image(image, image_root, mask_root)
        for image in images
    }
    actual_masks = {
        path
        for path in mask_root.rglob("*.png")
        if path.is_file()
    }

    missing_masks = sorted(expected_masks - actual_masks)
    extra_masks = sorted(actual_masks - expected_masks)

    if missing_masks:
        message = "\n".join(f"  {path}" for path in missing_masks[:20])
        raise RuntimeError(
            f"{split} 有 {len(missing_masks)} 张图片缺少 mask：\n{message}"
        )

    if extra_masks:
        message = "\n".join(f"  {path}" for path in extra_masks[:20])
        raise RuntimeError(
            f"{split} 有 {len(extra_masks)} 个多余 mask：\n{message}"
        )

    pixel_counts: Counter[int] = Counter()

    for index, image_path in enumerate(images, start=1):
        mask_path = mask_path_for_image(image_path, image_root, mask_root)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {image_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"无法读取 mask: {mask_path}")

        if mask.ndim != 2:
            raise RuntimeError(
                f"mask 必须是单通道 PNG: {mask_path}, shape={mask.shape}"
            )

        if not np.issubdtype(mask.dtype, np.integer):
            raise RuntimeError(
                f"mask 必须是整数类型: {mask_path}, dtype={mask.dtype}"
            )

        image_h, image_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]
        if (image_h, image_w) != (mask_h, mask_w):
            raise RuntimeError(
                "图片和 mask 尺寸不一致：\n"
                f"  image: {image_path} -> {image_w}x{image_h}\n"
                f"  mask : {mask_path} -> {mask_w}x{mask_h}"
            )

        unique_values, counts = np.unique(mask, return_counts=True)
        unique_ids = {int(value) for value in unique_values}
        invalid_ids = sorted(unique_ids - VALID_CLASS_IDS)

        if invalid_ids:
            raise RuntimeError(
                f"mask 存在非法类别值: {mask_path}, invalid={invalid_ids}"
            )

        for class_id, count in zip(unique_values, counts):
            pixel_counts[int(class_id)] += int(count)

        if index == 1 or index % 500 == 0 or index == len(images):
            print(f"  [{split}] checked {index}/{len(images)}")

    return len(images), pixel_counts


def print_class_statistics(
    split: str,
    image_count: int,
    pixel_counts: Counter[int],
) -> None:
    total_pixels = sum(pixel_counts.values())

    print(f"{split}: {image_count} images")
    for class_id, class_name in EXPECTED_NAMES.items():
        count = pixel_counts.get(class_id, 0)
        ratio = count / total_pixels * 100.0 if total_pixels else 0.0
        print(
            f"  {class_id}: {class_name:<16} "
            f"{count:>14d} px  {ratio:>7.3f}%"
        )


def check_environment() -> None:
    print("=" * 88)
    print("YOLO26 semantic segmentation - round 2 formal training")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(DEVICE)}")

    print(f"Model       : {MODEL_PATH}")
    print(f"Dataset     : {DATASET_ROOT}")
    print(f"Data YAML   : {DATA_YAML}")
    print(f"Run output  : {RUN_DIR}")
    print(f"Best export : {EXPORTED_BEST}")
    print("=" * 88)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"找不到上一轮权重: {MODEL_PATH}")

    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"找不到整理后的数据集: {DATASET_ROOT}")

    if RUN_DIR.exists():
        raise FileExistsError(
            f"训练输出目录已经存在: {RUN_DIR}\n"
            "为避免混入旧结果，请删除该目录，或修改脚本中的 RUN_NAME。"
        )

    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "脚本要求使用 CUDA，但 torch.cuda.is_available() 为 False。"
        )

    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE 必须大于 0")

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTED_BEST.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    try:
        check_environment()
        load_dataset_yaml()

        print("\n开始检查训练集和验证集……")
        train_count, train_pixels = validate_split("train")
        val_count, val_pixels = validate_split("val")

        print("\n数据集检查通过")
        print("=" * 88)
        print_class_statistics("Train", train_count, train_pixels)
        print("-" * 88)
        print_class_statistics("Val", val_count, val_pixels)
        print("=" * 88)

        model = YOLO(str(MODEL_PATH), task="semantic")

        model_names = normalize_names(model.names)
        if model_names != EXPECTED_NAMES:
            print("[WARN] 071801.pt 内保存的是旧的错误类别文字。")
            print(f"       checkpoint: {model_names}")
            print(f"       corrected : {EXPECTED_NAMES}")
            print("       只修正名称元数据，不重排输出通道和标签数字。")
            model.model.names = EXPECTED_NAMES.copy()

        print(f"Detected task: {model.task}")
        print("训练使用的正确类别映射：")
        for class_id, class_name in EXPECTED_NAMES.items():
            print(f"  {class_id}: {class_name}")

        print("\n开始正式训练……")
        results = model.train(**TRAIN_ARGS)
        _ = results

        best_path = RUN_DIR / "weights" / "best.pt"
        last_path = RUN_DIR / "weights" / "last.pt"

        if not best_path.is_file():
            raise RuntimeError(f"训练结束但没有找到 best.pt: {best_path}")

        shutil.copy2(best_path, EXPORTED_BEST)

        print("\n" + "=" * 88)
        print("训练完成")
        print(f"Run directory : {RUN_DIR}")
        print(f"Best weights  : {best_path}")
        print(f"Last weights  : {last_path}")
        print(f"Best copy     : {EXPORTED_BEST}")
        print("=" * 88)

        return 0

    except KeyboardInterrupt:
        print(
            "\n[STOPPED] 训练被用户中断。可使用 runs 目录中的 last.pt "
            "另行编写恢复训练配置。",
            file=sys.stderr,
        )
        return 130

    except torch.cuda.OutOfMemoryError:
        print(
            "\n[CUDA OOM] 显存不足。把脚本中的 BATCH_SIZE 从 8 改为 4 后重试。",
            file=sys.stderr,
        )
        return 2

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
