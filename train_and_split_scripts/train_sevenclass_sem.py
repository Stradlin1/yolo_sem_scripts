#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


WORKSPACE = Path("/home/xhm/Desktop/final_train")
DATASET_ROOT = WORKSPACE / "datasets"
DATA_YAML = DATASET_ROOT / "data_seven.yaml"

IMAGES_ROOT = DATASET_ROOT / "images"
MASKS_ROOT = DATASET_ROOT / "corrected_seven"

# 可改成你已经训练很好的六类 best.pt。
MODEL_PATH = Path("/home/xhm/Desktop/sem_train/models/yolo26n-sem.pt")

PROJECT_DIR = WORKSPACE / "runs" / "sevenclass_sem"
RUN_NAME = "yolo26n_sem_7class"

EXPECTED_RUNS = [
    "2026-06-28_01-05-54",
    "2026-06-28_01-08-45",
    "2026-06-28_01-13-14",
    "2026-06-28_01-18-35",
    "2026-06-28_01-25-30",
]

CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
    6: "Bucket",
}

IMGSZ = 320
EPOCHS = 600
PATIENCE = 30
BATCH = 4
DEVICE: int | str = 0
WORKERS = 8
SEED = 42
AMP = False

OPTIMIZER = "AdamW"
LR0 = 1.0e-3
WEIGHT_DECAY = 5.0e-4
EXIST_OK = False
VALIDATE_DATASET = True

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VALID_CLASS_IDS = set(CLASS_NAMES)


def write_data_yaml() -> None:
    config: dict[str, Any] = {
        "path": str(DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "corrected_seven",
        "nc": 7,
        "names": CLASS_NAMES,
    }

    if (IMAGES_ROOT / "test").is_dir() and (MASKS_ROOT / "test").is_dir():
        config["test"] = "images/test"

    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    with DATA_YAML.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    print(f"[OK] 已生成：{DATA_YAML}")


def iter_images(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def check_expected_runs(split: str) -> None:
    image_split_root = IMAGES_ROOT / split
    mask_split_root = MASKS_ROOT / split

    if not image_split_root.is_dir():
        raise FileNotFoundError(f"缺少图片目录：{image_split_root}")

    if not mask_split_root.is_dir():
        raise FileNotFoundError(f"缺少标签目录：{mask_split_root}")

    missing_images = [
        run for run in EXPECTED_RUNS
        if not (image_split_root / run).is_dir()
    ]
    missing_masks = [
        run for run in EXPECTED_RUNS
        if not (mask_split_root / run).is_dir()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{split} 缺少图片 run：{missing_images}"
        )

    if missing_masks:
        raise FileNotFoundError(
            f"{split} 缺少标签 run：{missing_masks}"
        )


def validate_split(split: str) -> dict[str, Any]:
    image_root = IMAGES_ROOT / split
    mask_root = MASKS_ROOT / split

    image_paths = iter_images(image_root)
    if not image_paths:
        raise RuntimeError(f"{split} 中没有图片：{image_root}")

    missing_masks: list[Path] = []
    class_pixels: Counter[int] = Counter()
    images_with_bucket = 0
    total_pixels = 0

    expected_mask_paths: set[Path] = set()

    for index, image_path in enumerate(image_paths, start=1):
        relative_image = image_path.relative_to(image_root)
        mask_path = mask_root / relative_image.with_suffix(".png")
        expected_mask_paths.add(mask_path)

        if not mask_path.is_file():
            missing_masks.append(mask_path)
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片：{image_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"无法读取标签：{mask_path}")

        if mask.ndim != 2:
            raise RuntimeError(
                f"mask 必须是单通道：{mask_path}，shape={mask.shape}"
            )

        if mask.dtype != np.uint8:
            raise RuntimeError(
                f"mask 必须是 uint8：{mask_path}，dtype={mask.dtype}"
            )

        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"图片和 mask 尺寸不一致：\n"
                f"图片：{image_path}，shape={image.shape[:2]}\n"
                f"标签：{mask_path}，shape={mask.shape}"
            )

        ids, counts = np.unique(mask, return_counts=True)
        ids_set = set(ids.astype(int).tolist())
        illegal_ids = ids_set - VALID_CLASS_IDS

        if illegal_ids:
            raise RuntimeError(
                f"标签包含非法类别：{mask_path}\n"
                f"实际 ID={sorted(ids_set)}\n"
                f"允许 ID={sorted(VALID_CLASS_IDS)}"
            )

        for class_id, count in zip(ids, counts):
            class_pixels[int(class_id)] += int(count)

        if np.any(mask == 6):
            images_with_bucket += 1

        total_pixels += int(mask.size)

        if index == 1 or index % 500 == 0 or index == len(image_paths):
            print(
                f"[检查 {split}] {index}/{len(image_paths)} "
                f"{relative_image.as_posix()}"
            )

    if missing_masks:
        preview = "\n".join(str(path) for path in missing_masks[:20])
        raise RuntimeError(
            f"{split} 有 {len(missing_masks)} 张图片缺少标签：\n{preview}"
        )

    actual_masks = set(mask_root.rglob("*.png"))
    extra_masks = sorted(actual_masks - expected_mask_paths)

    if extra_masks:
        preview = "\n".join(str(path) for path in extra_masks[:20])
        raise RuntimeError(
            f"{split} 有 {len(extra_masks)} 个多余标签：\n{preview}"
        )

    return {
        "images": len(image_paths),
        "images_with_bucket": images_with_bucket,
        "class_pixels": class_pixels,
        "total_pixels": total_pixels,
    }


def print_split_summary(split: str, stats: dict[str, Any]) -> None:
    print("-" * 80)
    print(f"split                : {split}")
    print(f"图片数量             : {stats['images']}")
    print(f"包含 Bucket 的图片数 : {stats['images_with_bucket']}")

    class_pixels: Counter[int] = stats["class_pixels"]
    total_pixels: int = stats["total_pixels"]

    for class_id in range(7):
        count = int(class_pixels.get(class_id, 0))
        ratio = count / total_pixels if total_pixels else 0.0

        print(
            f"class {class_id} {CLASS_NAMES[class_id]:<20} "
            f"pixels={count:<12} ratio={ratio:.6%}"
        )

    if int(class_pixels.get(6, 0)) == 0:
        raise RuntimeError(
            f"{split} 中没有 Bucket 像素，无法训练或验证 class 6。"
        )


def build_train_kwargs() -> dict[str, Any]:
    return {
        "data": str(DATA_YAML),
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "patience": PATIENCE,
        "batch": BATCH,
        "device": DEVICE,
        "workers": WORKERS,
        "seed": SEED,
        "deterministic": True,
        "amp": AMP,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "weight_decay": WEIGHT_DECAY,
        "project": str(PROJECT_DIR),
        "name": RUN_NAME,
        "exist_ok": EXIST_OK,
        "resume": False,
        "pretrained": True,
        "val": True,
        "plots": True,
        "save": True,
        "cache": False,

        "fliplr": 0.0,
        "flipud": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,

        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }


def main() -> None:
    print("=" * 80)
    print("7 类 BEV 语义分割训练")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"Dataset      : {DATASET_ROOT}")
    print(f"Images       : {IMAGES_ROOT}")
    print(f"Masks        : {MASKS_ROOT}")
    print(f"Model        : {MODEL_PATH}")
    print(f"Project      : {PROJECT_DIR}")
    print(f"Run name     : {RUN_NAME}")
    print(f"Classes      : {CLASS_NAMES}")
    print("=" * 80)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"模型权重不存在：{MODEL_PATH}\n"
            "请修改脚本顶部的 MODEL_PATH。"
        )

    check_expected_runs("train")
    check_expected_runs("val")

    write_data_yaml()

    if VALIDATE_DATASET:
        train_stats = validate_split("train")
        val_stats = validate_split("val")

        print_split_summary("train", train_stats)
        print_split_summary("val", val_stats)

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "当前环境没有安装 ultralytics，请在 yolo_sem 环境中运行。"
        ) from error

    print("-" * 80)
    print(f"Ultralytics  : {ultralytics.__version__}")

    model = YOLO(str(MODEL_PATH))

    print(f"Model task   : {getattr(model, 'task', 'unknown')}")
    print("=" * 80)
    print("开始训练。")
    print("类别数量已经变成 7，因此固定 resume=False。")
    print("background=5，Bucket=6。")
    print("=" * 80)

    results = model.train(**build_train_kwargs())

    print("=" * 80)
    print("训练结束")
    print(f"结果目录：{PROJECT_DIR / RUN_NAME}")
    print(
        f"最佳权重："
        f"{PROJECT_DIR / RUN_NAME / 'weights' / 'best.pt'}"
    )
    print(
        f"最后权重："
        f"{PROJECT_DIR / RUN_NAME / 'weights' / 'last.pt'}"
    )
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
