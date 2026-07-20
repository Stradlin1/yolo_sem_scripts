#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

VALID_CLASS_IDS = set(range(6))

CLASS_NAMES = {
    0: "background",
    1: "P",
    2: "A_DaTing",
    3: "B_TongDao",
    4: "C_ZhenLiaoShi",
    5: "JinQu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare train/val data by preferring corrected masks and "
            "falling back to ONNX autolabel masks."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/xhm/Desktop/sem_train"),
    )
    parser.add_argument(
        "--run-name",
        default="2026-06-28_01-08-45",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--output-name",
        default="bev_sem_verify",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def find_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def validate_pair(
    image_path: Path,
    mask_path: Path,
) -> tuple[list[int], int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"无法读取 mask: {mask_path}")

    if mask.ndim != 2:
        raise RuntimeError(
            f"mask 不是单通道: {mask_path}, shape={mask.shape}"
        )

    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]

    if (image_h, image_w) != (mask_h, mask_w):
        raise RuntimeError(
            "图片和 mask 尺寸不一致:\n"
            f"  image: {image_path} -> {image_w}x{image_h}\n"
            f"  mask : {mask_path} -> {mask_w}x{mask_h}"
        )

    mask_ids = set(int(value) for value in np.unique(mask))
    invalid_ids = mask_ids - VALID_CLASS_IDS

    if invalid_ids:
        raise RuntimeError(
            f"mask 存在非法类别值: {mask_path}, "
            f"invalid={sorted(invalid_ids)}"
        )

    return sorted(mask_ids), image_w, image_h


def copy_pair(
    image_path: Path,
    mask_path: Path,
    relative_path: Path,
    output_root: Path,
    split: str,
) -> tuple[Path, Path]:
    image_output = output_root / "images" / split / relative_path
    mask_output = (
        output_root / "masks" / split / relative_path
    ).with_suffix(".png")

    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image_path, image_output)
    shutil.copy2(mask_path, mask_output)

    return image_output, mask_output


def main() -> int:
    args = parse_args()

    if not 0.0 < args.train_ratio < 1.0:
        print("[ERROR] --train-ratio 必须在 0 和 1 之间", file=sys.stderr)
        return 1

    root = args.root.expanduser().resolve()
    datasets_root = root / "datasets"

    image_root = datasets_root / "bev_all" / args.run_name
    autolabel_root = (
        datasets_root
        / "seg_autolabel"
        / args.run_name
        / "masks_id"
    )
    corrected_root = (
        datasets_root
        / "seg_corrected"
        / args.run_name
        / "masks"
    )
    output_root = datasets_root / args.output_name

    if not image_root.exists():
        print(f"[ERROR] 图片目录不存在: {image_root}", file=sys.stderr)
        return 1

    if not autolabel_root.exists():
        print(f"[ERROR] 自动标签目录不存在: {autolabel_root}", file=sys.stderr)
        return 1

    images = find_images(image_root)
    if len(images) < 2:
        print(f"[ERROR] 图片数量不足: {len(images)}", file=sys.stderr)
        return 1

    pairs = []
    missing = []
    corrected_count = 0
    autolabel_count = 0

    for image_path in images:
        relative_path = image_path.relative_to(image_root)

        corrected_mask = (
            corrected_root / relative_path
        ).with_suffix(".png")
        autolabel_mask = (
            autolabel_root / relative_path
        ).with_suffix(".png")

        if corrected_mask.exists():
            mask_path = corrected_mask
            source = "corrected"
            corrected_count += 1
        elif autolabel_mask.exists():
            mask_path = autolabel_mask
            source = "autolabel"
            autolabel_count += 1
        else:
            missing.append((image_path, corrected_mask, autolabel_mask))
            continue

        class_ids, width, height = validate_pair(image_path, mask_path)

        pairs.append({
            "image_path": image_path,
            "relative_path": relative_path,
            "mask_path": mask_path,
            "source": source,
            "class_ids": class_ids,
            "width": width,
            "height": height,
        })

    print("=" * 78)
    print(f"BEV images      : {image_root}")
    print(f"Autolabel masks : {autolabel_root}")
    print(f"Corrected masks : {corrected_root}")
    print(f"Output dataset  : {output_root}")
    print("-" * 78)
    print(f"Found images    : {len(images)}")
    print(f"Corrected used  : {corrected_count}")
    print(f"Autolabel used  : {autolabel_count}")
    print(f"Missing both    : {len(missing)}")
    print("=" * 78)

    if missing:
        print(
            "[ERROR] 以下图片既没有修正 mask，也没有自动 mask：",
            file=sys.stderr,
        )
        for image_path, corrected_mask, autolabel_mask in missing[:20]:
            print(f"  image     : {image_path}", file=sys.stderr)
            print(f"  corrected : {corrected_mask}", file=sys.stderr)
            print(f"  autolabel : {autolabel_mask}", file=sys.stderr)
        return 2

    split_index = int(len(pairs) * args.train_ratio)
    split_index = max(1, min(split_index, len(pairs) - 1))

    train_pairs = pairs[:split_index]
    val_pairs = pairs[split_index:]

    if output_root.exists():
        if not args.overwrite:
            print(
                f"[ERROR] 输出目录已存在: {output_root}\n"
                "请加 --overwrite 重新生成。",
                file=sys.stderr,
            )
            return 3
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    records = []

    for split, split_pairs in (
        ("train", train_pairs),
        ("val", val_pairs),
    ):
        for pair in split_pairs:
            image_output, mask_output = copy_pair(
                image_path=pair["image_path"],
                mask_path=pair["mask_path"],
                relative_path=pair["relative_path"],
                output_root=output_root,
                split=split,
            )

            records.append({
                "split": split,
                "relative_path": str(pair["relative_path"]),
                "mask_source": pair["source"],
                "class_ids": ",".join(map(str, pair["class_ids"])),
                "width": pair["width"],
                "height": pair["height"],
                "source_image": str(pair["image_path"]),
                "source_mask": str(pair["mask_path"]),
                "output_image": str(image_output),
                "output_mask": str(mask_output),
            })

    data_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": CLASS_NAMES,
    }

    yaml_path = output_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_yaml,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "split",
            "relative_path",
            "mask_source",
            "class_ids",
            "width",
            "height",
            "source_image",
            "source_mask",
            "output_image",
            "output_mask",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    train_corrected = sum(
        pair["source"] == "corrected"
        for pair in train_pairs
    )
    val_corrected = sum(
        pair["source"] == "corrected"
        for pair in val_pairs
    )

    print()
    print("数据集建立完成")
    print("=" * 78)
    print(
        f"Train: {len(train_pairs)} "
        f"(corrected={train_corrected}, "
        f"autolabel={len(train_pairs) - train_corrected})"
    )
    print(
        f"Val  : {len(val_pairs)} "
        f"(corrected={val_corrected}, "
        f"autolabel={len(val_pairs) - val_corrected})"
    )
    print(f"YAML     : {yaml_path}")
    print(f"Manifest : {manifest_path}")
    print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
