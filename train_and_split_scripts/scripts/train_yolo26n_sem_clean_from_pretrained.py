#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
从本地 YOLO26n-sem 预训练权重重新开始训练。

目的：
- 不加载之前掺杂镜像图片训练出的 071801.pt、071802.pt 或其他 best.pt；
- 只使用本地原始 YOLO26n-sem 预训练权重；
- 使用清理后的 bev_sem_round2_merged 数据集；
- 2026-06-28_01-30-09 只保留在 test，不进入 train/val；
- 不启用会改变 BEV 几何方向的增强；
- 不联网下载任何模型权重。

正确类别映射：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background
"""

from __future__ import annotations

import argparse
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
# 默认路径
# =============================================================================

WORKSPACE = Path("/home/xhm/Desktop/sem_train")

# 必须是本地原始语义分割预训练权重，不能指向之前训练出的 best.pt。
DEFAULT_BASE_MODEL = WORKSPACE / "models" / "yolo26n-sem.pt"

DATASET_ROOT = WORKSPACE / "datasets" / "bev_sem_round2_merged"
DATA_YAML = DATASET_ROOT / "data.yaml"

PROJECT_DIR = WORKSPACE / "runs"
DEFAULT_RUN_NAME = "clean_from_yolo26n_sem_pretrained"

# 训练完成后额外复制一份最佳权重。
EXPORT_DIR = WORKSPACE / "models"


EXPECTED_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

TEST_ONLY_RUNS = {
    "2026-06-28_01-30-09",
}

VALID_CLASS_IDS = set(EXPECTED_NAMES)

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train clean BEV semantic model from local pretrained weights."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_BASE_MODEL,
        help=(
            "本地 YOLO26n-sem 预训练权重。"
            f"默认: {DEFAULT_BASE_MODEL}"
        ),
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"训练输出名称。默认: {DEFAULT_RUN_NAME}",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=800,
        help="训练轮数。默认: 800",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="批大小。关闭 AMP 后默认使用 4。",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="训练设备。默认: 0",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="数据加载线程。默认: 8",
    )
    return parser.parse_args()


def normalize_names(raw_names: object) -> dict[int, str]:
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

    raise TypeError(
        f"不支持的类别 names 类型: {type(raw_names).__name__}"
    )


def list_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def contains_run(path: Path, run_name: str) -> bool:
    return any(run_name in part for part in path.parts)


def mask_for_image(
    image_path: Path,
    image_root: Path,
    mask_root: Path,
) -> Path:
    relative_path = image_path.relative_to(image_root)
    return (mask_root / relative_path).with_suffix(".png")


def load_and_validate_yaml() -> dict:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"找不到 data.yaml: {DATA_YAML}")

    with DATA_YAML.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise RuntimeError(f"data.yaml 内容无效: {DATA_YAML}")

    names = normalize_names(data.get("names"))
    if names != EXPECTED_NAMES:
        raise RuntimeError(
            "data.yaml 类别顺序错误。\n"
            f"实际: {names}\n"
            f"要求: {EXPECTED_NAMES}"
        )

    required_paths = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }

    for key, expected in required_paths.items():
        actual = str(data.get(key, ""))
        if actual != expected:
            raise RuntimeError(
                f"data.yaml 的 {key} 配置错误: "
                f"actual={actual!r}, expected={expected!r}"
            )

    masks_dir = str(data.get("masks_dir", ""))
    if masks_dir != "masks":
        raise RuntimeError(
            f"data.yaml 的 masks_dir 应为 'masks'，实际为 {masks_dir!r}"
        )

    return data


def validate_split(
    split: str,
    require_test_run: bool,
) -> tuple[int, Counter[int]]:
    image_root = DATASET_ROOT / "images" / split
    mask_root = DATASET_ROOT / "masks" / split

    if not image_root.is_dir():
        raise FileNotFoundError(f"缺少图片目录: {image_root}")

    if not mask_root.is_dir():
        raise FileNotFoundError(f"缺少 mask 目录: {mask_root}")

    images = list_images(image_root)
    if not images:
        raise RuntimeError(f"{split} 中没有图片: {image_root}")

    test_run_hits = {
        run_name: 0
        for run_name in TEST_ONLY_RUNS
    }

    expected_masks: set[Path] = set()
    class_pixel_counts: Counter[int] = Counter()

    for index, image_path in enumerate(images, start=1):
        for run_name in TEST_ONLY_RUNS:
            if contains_run(image_path.relative_to(image_root), run_name):
                test_run_hits[run_name] += 1

        mask_path = mask_for_image(
            image_path=image_path,
            image_root=image_root,
            mask_root=mask_root,
        )
        expected_masks.add(mask_path)

        if not mask_path.is_file():
            raise RuntimeError(
                "图片缺少对应 mask:\n"
                f"  image: {image_path}\n"
                f"  mask : {mask_path}"
            )

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
                f"mask 必须保存整数类别 ID: "
                f"{mask_path}, dtype={mask.dtype}"
            )

        image_h, image_w = image.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if (image_h, image_w) != (mask_h, mask_w):
            raise RuntimeError(
                "图片与 mask 尺寸不一致:\n"
                f"  image: {image_path} -> {image_w}x{image_h}\n"
                f"  mask : {mask_path} -> {mask_w}x{mask_h}"
            )

        class_ids, counts = np.unique(mask, return_counts=True)
        invalid_ids = sorted(
            {int(value) for value in class_ids} - VALID_CLASS_IDS
        )

        if invalid_ids:
            raise RuntimeError(
                f"mask 含非法类别值: {mask_path}, invalid={invalid_ids}"
            )

        for class_id, count in zip(class_ids, counts):
            class_pixel_counts[int(class_id)] += int(count)

        if index == 1 or index % 500 == 0 or index == len(images):
            print(f"  [{split}] checked {index}/{len(images)}")

    actual_masks = {
        path
        for path in mask_root.rglob("*.png")
        if path.is_file()
    }
    extra_masks = sorted(actual_masks - expected_masks)

    if extra_masks:
        lines = "\n".join(f"  {path}" for path in extra_masks[:20])
        raise RuntimeError(
            f"{split} 中存在 {len(extra_masks)} 个多余 mask:\n{lines}"
        )

    if require_test_run:
        for run_name, count in test_run_hits.items():
            if count == 0:
                raise RuntimeError(
                    f"测试集未发现指定保留 run: {run_name}"
                )
    else:
        leaked_runs = {
            run_name: count
            for run_name, count in test_run_hits.items()
            if count > 0
        }
        if leaked_runs:
            raise RuntimeError(
                f"{split} 中混入了测试专用 run: {leaked_runs}"
            )

    return len(images), class_pixel_counts


def print_class_statistics(
    split: str,
    image_count: int,
    pixel_counts: Counter[int],
) -> None:
    total_pixels = sum(pixel_counts.values())

    print(f"{split}: {image_count} images")
    for class_id, class_name in EXPECTED_NAMES.items():
        pixel_count = pixel_counts.get(class_id, 0)
        ratio = (
            pixel_count / total_pixels * 100.0
            if total_pixels
            else 0.0
        )
        print(
            f"  {class_id}: {class_name:<16} "
            f"{pixel_count:>14d} px  {ratio:>7.3f}%"
        )


def validate_base_model(model_path: Path) -> YOLO:
    if not model_path.is_file():
        raise FileNotFoundError(
            "找不到本地预训练权重:\n"
            f"  {model_path}\n\n"
            "请把原始 yolo26n-sem.pt 放到该位置，"
            "或运行脚本时使用 --model 指定本地路径。"
        )

    forbidden_names = {
        "071801.pt",
        "071802.pt",
        "best.pt",
        "last.pt",
    }

    if model_path.name.lower() in forbidden_names:
        raise RuntimeError(
            "当前 --model 看起来是之前训练出的权重，"
            "不能用于本次从原始预训练权重重新开始:\n"
            f"  {model_path}"
        )

    model = YOLO(str(model_path), task="semantic")

    if str(model.task) != "semantic":
        raise RuntimeError(
            f"本地预训练模型 task 不是 semantic: {model.task}"
        )

    return model


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    run_name = args.run_name.strip()
    run_dir = PROJECT_DIR / run_name
    exported_best = EXPORT_DIR / f"{run_name}_best.pt"

    try:
        if args.epochs < 1:
            raise ValueError("--epochs 必须大于 0")

        if args.batch < 1:
            raise ValueError("--batch 必须大于 0")

        if not run_name:
            raise ValueError("--run-name 不能为空")

        if run_dir.exists():
            raise FileExistsError(
                f"训练目录已经存在: {run_dir}\n"
                "为防止混入旧结果，请删除它或更换 --run-name。"
            )

        if args.device != "cpu" and not torch.cuda.is_available():
            raise RuntimeError(
                "要求使用 CUDA，但 torch.cuda.is_available() 为 False。"
            )

        print("=" * 96)
        print("Clean BEV semantic training from local pretrained weights")
        print(f"Ultralytics : {ultralytics.__version__}")
        print(f"PyTorch     : {torch.__version__}")
        print(f"CUDA        : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"GPU         : {torch.cuda.get_device_name(0)}")

        print(f"Base model  : {model_path}")
        print(f"Dataset     : {DATASET_ROOT}")
        print(f"Data YAML   : {DATA_YAML}")
        print(f"Run output  : {run_dir}")
        print(f"Best export : {exported_best}")
        print("=" * 96)

        load_and_validate_yaml()

        print("\n检查 train/val/test 数据……")
        train_count, train_pixels = validate_split(
            split="train",
            require_test_run=False,
        )
        val_count, val_pixels = validate_split(
            split="val",
            require_test_run=False,
        )
        test_count, test_pixels = validate_split(
            split="test",
            require_test_run=True,
        )

        print("\n数据集检查通过")
        print("=" * 96)
        print_class_statistics("Train", train_count, train_pixels)
        print("-" * 96)
        print_class_statistics("Val", val_count, val_pixels)
        print("-" * 96)
        print_class_statistics("Test (not used for training)", test_count, test_pixels)
        print("=" * 96)

        model = validate_base_model(model_path)

        print(f"Loaded task : {model.task}")
        print(f"Base names  : {normalize_names(model.names)}")
        print("Dataset names:")
        for class_id, class_name in EXPECTED_NAMES.items():
            print(f"  {class_id}: {class_name}")

        PROJECT_DIR.mkdir(parents=True, exist_ok=True)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        train_args = {
            "data": str(DATA_YAML),
            "imgsz": 320,
            "epochs": args.epochs,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,

            "project": str(PROJECT_DIR),
            "name": run_name,
            "exist_ok": False,

            # 明确从本地原始预训练权重开始，不加载旧训练 checkpoint。
            "pretrained": str(model_path),
            "resume": False,

            "optimizer": "auto",
            "seed": 42,
            "deterministic": True,
            "patience": 100,

            "save": True,
            "save_period": 10,
            "plots": True,
            "verbose": True,
            "cache": False,
            "val": True,

            # 关闭 AMP，避免训练前额外拉取用于 AMP 检查的模型。
            "amp": False,

            # 固定输入，不做多尺度训练。
            "multi_scale": 0.0,

            # BEV 方向有物理含义，禁止几何和拼接增强。
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
            "close_mosaic": 0,

            # 保持颜色输入不变。
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
        }

        print("\n开始训练")
        print("=" * 96)
        print(f"Pretrained source : {model_path}")
        print("Previous checkpoints are not used.")
        print(f"Epochs            : {args.epochs}")
        print(f"Batch             : {args.batch}")
        print("Image size        : 320")
        print("AMP               : False")
        print("Test split        : excluded from train/val")
        print("=" * 96)

        model.train(**train_args)

        best_path = run_dir / "weights" / "best.pt"
        last_path = run_dir / "weights" / "last.pt"

        if not best_path.is_file():
            raise RuntimeError(
                f"训练完成但没有找到 best.pt: {best_path}"
            )

        shutil.copy2(best_path, exported_best)

        print("\n训练完成")
        print("=" * 96)
        print(f"Run directory : {run_dir}")
        print(f"Best weights  : {best_path}")
        print(f"Last weights  : {last_path}")
        print(f"Best copy     : {exported_best}")
        print("=" * 96)

        return 0

    except KeyboardInterrupt:
        print("\n[STOPPED] 训练被用户中断。", file=sys.stderr)
        return 130

    except torch.cuda.OutOfMemoryError:
        print(
            "\n[CUDA OOM] 显存不足。把 --batch 改为 2 后重新运行。",
            file=sys.stderr,
        )
        return 2

    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
