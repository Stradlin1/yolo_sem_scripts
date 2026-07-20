#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare the second-round dense semantic-segmentation dataset.

Processing order:
1. Organize each new seg_pt run:
   - Prefer masks_corrected for images that were manually saved.
   - Fall back to masks_id for images that were not manually saved.
   - Require every image to have at least one valid mask.
2. Merge the organized new samples from all requested runs.
3. Merge them with the existing bev_sem_verify dataset.
4. Sort all unique samples by run timestamp and natural image filename order.
5. Put every 10th sample into validation; put the other 9 into training.

Final output layout:

    <output>/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── masks/
    │   ├── train/
    │   └── val/
    ├── data.yaml
    └── manifest.csv

Semantic mask convention:
    0 = background
    1 = P
    2 = A_DaTing
    3 = B_TongDao
    4 = C_ZhenLiaoShi
    5 = JinQu
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp",
}

VALID_CLASS_IDS = set(range(6))

CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

RUN_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
NATURAL_PATTERN = re.compile(r"(\d+)")


@dataclass
class Pair:
    image_path: Path
    mask_path: Path
    logical_rel: Path
    run_name: str
    source_group: str
    mask_source: str
    source_detail: str
    class_ids: list[int] | None = None
    width: int | None = None
    height: int | None = None

    @property
    def duplicate_key(self) -> str:
        """Stable sample key independent of image extension."""
        return self.logical_rel.with_suffix("").as_posix().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Organize new seg_pt labels, merge them with bev_sem_verify, "
            "and put every 10th chronological sample into validation."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/xhm/Desktop/sem_train"),
        help="sem_train root directory.",
    )
    parser.add_argument(
        "--previous-dataset",
        type=Path,
        default=None,
        help=(
            "Existing prepared dataset. Default: "
            "<root>/datasets/bev_sem_verify"
        ),
    )
    parser.add_argument(
        "--seg-pt-root",
        type=Path,
        default=None,
        help="Copied seg_pt root. Default: <root>/datasets/seg_pt",
    )
    parser.add_argument(
        "--bev-root",
        type=Path,
        default=Path("/home/xhm/Desktop/relocate_ws/data/extracted/bev_all"),
        help=(
            "Fallback BEV image root. The script first checks "
            "<root>/datasets/bev_all, then this path."
        ),
    )
    parser.add_argument(
        "--new-runs",
        nargs="+",
        default=[
            "2026-06-28_01-05-54_071801",
            "2026-06-28_01-13-14_071801",
        ],
        help="Subdirectories under seg_pt to merge.",
    )
    parser.add_argument(
        "--corrected-dir-name",
        default="masks_corrected",
        help="Manual-mask directory name inside each seg_pt run.",
    )
    parser.add_argument(
        "--auto-dir-name",
        default="masks_id",
        help="Automatic-mask directory name inside each seg_pt run.",
    )
    parser.add_argument(
        "--output-name",
        default="bev_sem_round2_merged",
        help="Final dataset directory name under <root>/datasets.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help=(
            "After chronological sorting, every Nth sample is validation. "
            "Default: 10, so samples 10,20,30,... are validation."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the output dataset if it already exists.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def find_images(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ),
        key=lambda path: natural_tokens(path.relative_to(root).as_posix()),
    )


def natural_tokens(text: str) -> tuple[object, ...]:
    parts = NATURAL_PATTERN.split(text.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def extract_run_name(value: str | Path) -> str:
    match = RUN_PATTERN.search(str(value))
    return match.group(0) if match else ""


def strip_run_suffix(run_dir_name: str) -> str:
    run_name = extract_run_name(run_dir_name)
    if not run_name:
        raise RuntimeError(
            "无法从新数据目录名提取时间戳: "
            f"{run_dir_name}. 需要包含 YYYY-MM-DD_HH-MM-SS。"
        )
    return run_name


def path_after_run(path_value: str | Path, run_name: str) -> Path | None:
    if not run_name:
        return None

    path = Path(str(path_value))
    parts = list(path.parts)
    for index, part in enumerate(parts):
        if part == run_name or run_name in part:
            tail = parts[index + 1:]
            return Path(*tail) if tail else None
    return None


def pair_sort_key(pair: Pair) -> tuple[object, ...]:
    try:
        run_time = datetime.strptime(pair.run_name, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        run_time = datetime.min

    return (
        run_time,
        natural_tokens(pair.logical_rel.as_posix()),
    )


def load_previous_manifest(previous_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    manifest_path = previous_root / "manifest.csv"
    rows: dict[tuple[str, str], dict[str, str]] = {}

    if not manifest_path.is_file():
        return rows

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = (row.get("split") or "").strip()
            relative = (
                row.get("relative_path")
                or row.get("logical_path")
                or ""
            ).strip()
            if split and relative:
                rows[(split, Path(relative).as_posix())] = row

    return rows


def load_previous_pairs(previous_root: Path) -> list[Pair]:
    """Read all old train/val samples and forget their previous split."""
    pairs: list[Pair] = []
    manifest = load_previous_manifest(previous_root)

    for old_split in ("train", "val"):
        image_root = previous_root / "images" / old_split
        mask_root = previous_root / "masks" / old_split

        if not image_root.is_dir():
            raise RuntimeError(f"旧数据集图片目录不存在: {image_root}")
        if not mask_root.is_dir():
            raise RuntimeError(f"旧数据集 mask 目录不存在: {mask_root}")

        for image_path in find_images(image_root):
            relative_path = image_path.relative_to(image_root)
            mask_path = (mask_root / relative_path).with_suffix(".png")
            if not mask_path.is_file():
                raise RuntimeError(
                    "旧数据集中图片缺少对应 mask:\n"
                    f"  image: {image_path}\n"
                    f"  mask : {mask_path}"
                )

            row = manifest.get((old_split, relative_path.as_posix()), {})
            source_image = row.get("source_image", "")
            run_name = (
                (row.get("run_name") or "").strip()
                or extract_run_name(source_image)
                or extract_run_name(relative_path)
                or extract_run_name(image_path)
            )

            tail = path_after_run(source_image, run_name)
            if tail is None:
                tail = path_after_run(relative_path, run_name)
            if tail is None:
                tail = relative_path

            if run_name:
                logical_rel = Path(run_name) / tail
            else:
                # Keep unknown old samples isolated so they cannot collide with
                # timestamped new samples solely because filenames match.
                logical_rel = Path("previous_unknown") / relative_path

            pairs.append(Pair(
                image_path=image_path,
                mask_path=mask_path,
                logical_rel=logical_rel,
                run_name=run_name,
                source_group="previous",
                mask_source="previous_dataset",
                source_detail=f"{previous_root.name}/{old_split}",
            ))

    return pairs


def choose_new_image_root(
    sem_train_bev_root: Path,
    fallback_bev_root: Path,
    run_name: str,
) -> Path:
    candidates = [
        sem_train_bev_root / run_name,
        fallback_bev_root / run_name,
    ]

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        f"找不到 {run_name} 的 BEV 图片目录，已检查:\n{checked}"
    )


def select_new_mask(
    relative_path: Path,
    corrected_root: Path,
    auto_root: Path,
) -> tuple[Path, str]:
    corrected_mask = (corrected_root / relative_path).with_suffix(".png")
    auto_mask = (auto_root / relative_path).with_suffix(".png")

    if corrected_mask.is_file():
        return corrected_mask, "corrected"
    if auto_mask.is_file():
        return auto_mask, "autolabel"

    raise RuntimeError(
        "新数据图片同时缺少 corrected 和 autolabel mask:\n"
        f"  corrected: {corrected_mask}\n"
        f"  autolabel: {auto_mask}"
    )


def load_new_pairs(
    seg_pt_root: Path,
    sem_train_bev_root: Path,
    fallback_bev_root: Path,
    new_run_dirs: list[str],
    corrected_dir_name: str,
    auto_dir_name: str,
) -> list[Pair]:
    """
    First organize seg_pt: corrected masks override automatic masks.
    All new runs are then returned as one merged list.
    """
    pairs: list[Pair] = []

    for run_dir_name in new_run_dirs:
        run_name = strip_run_suffix(run_dir_name)
        run_root = seg_pt_root / run_dir_name
        corrected_root = run_root / corrected_dir_name
        auto_root = run_root / auto_dir_name

        if not run_root.is_dir():
            raise RuntimeError(f"新标注目录不存在: {run_root}")
        if not auto_root.is_dir():
            raise RuntimeError(f"自动标签目录不存在: {auto_root}")
        # masks_corrected may be absent when no image was manually saved.

        image_root = choose_new_image_root(
            sem_train_bev_root=sem_train_bev_root,
            fallback_bev_root=fallback_bev_root,
            run_name=run_name,
        )
        images = find_images(image_root)
        if not images:
            raise RuntimeError(f"新数据图片目录中没有图片: {image_root}")

        corrected_count = 0
        autolabel_count = 0

        for image_path in images:
            relative_path = image_path.relative_to(image_root)
            mask_path, mask_source = select_new_mask(
                relative_path=relative_path,
                corrected_root=corrected_root,
                auto_root=auto_root,
            )

            if mask_source == "corrected":
                corrected_count += 1
            else:
                autolabel_count += 1

            pairs.append(Pair(
                image_path=image_path,
                mask_path=mask_path,
                logical_rel=Path(run_name) / relative_path,
                run_name=run_name,
                source_group="new",
                mask_source=mask_source,
                source_detail=run_dir_name,
            ))

        print(
            f"[SEG_PT] {run_dir_name}: total={len(images)}, "
            f"corrected={corrected_count}, "
            f"autolabel_fallback={autolabel_count}"
        )

    return pairs


def validate_pair(pair: Pair) -> None:
    image = cv2.imread(str(pair.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片: {pair.image_path}")

    mask = cv2.imread(str(pair.mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"无法读取 mask: {pair.mask_path}")

    if mask.ndim == 3:
        channels_equal = (
            np.array_equal(mask[:, :, 0], mask[:, :, 1])
            and np.array_equal(mask[:, :, 1], mask[:, :, 2])
        )
        if not channels_equal:
            raise RuntimeError(
                f"mask 不是单通道类别图: {pair.mask_path}, shape={mask.shape}"
            )
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise RuntimeError(
            f"mask 维度错误: {pair.mask_path}, shape={mask.shape}"
        )
    if not np.issubdtype(mask.dtype, np.integer):
        raise RuntimeError(
            f"mask 必须保存整数类别值: {pair.mask_path}, dtype={mask.dtype}"
        )

    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]
    if (image_h, image_w) != (mask_h, mask_w):
        raise RuntimeError(
            "图片和 mask 尺寸不一致:\n"
            f"  image: {pair.image_path} -> {image_w}x{image_h}\n"
            f"  mask : {pair.mask_path} -> {mask_w}x{mask_h}"
        )

    class_ids = set(int(value) for value in np.unique(mask))
    invalid_ids = class_ids - VALID_CLASS_IDS
    if invalid_ids:
        raise RuntimeError(
            f"mask 存在非法类别值: {pair.mask_path}, "
            f"invalid={sorted(invalid_ids)}"
        )

    pair.class_ids = sorted(class_ids)
    pair.width = image_w
    pair.height = image_h


def merge_pairs(
    previous_pairs: list[Pair],
    new_pairs: list[Pair],
) -> tuple[list[Pair], int]:
    """New data replace old samples when both resolve to the same key."""
    merged: dict[str, Pair] = {}

    for pair in previous_pairs:
        key = pair.duplicate_key
        if key in merged:
            raise RuntimeError(
                "旧数据集内部存在重复样本键:\n"
                f"  key   : {key}\n"
                f"  first : {merged[key].image_path}\n"
                f"  second: {pair.image_path}"
            )
        merged[key] = pair

    replaced = 0
    for pair in new_pairs:
        key = pair.duplicate_key
        if key in merged:
            replaced += 1
            print(
                "[REPLACE] 新数据覆盖旧数据: "
                f"{key}\n"
                f"          old mask: {merged[key].mask_path}\n"
                f"          new mask: {pair.mask_path}"
            )
        merged[key] = pair

    return list(merged.values()), replaced


def split_every_n(
    ordered_pairs: list[Pair],
    val_every: int,
) -> tuple[list[Pair], list[Pair]]:
    """Use 1-based positions N, 2N, 3N, ... as validation samples."""
    train_pairs: list[Pair] = []
    val_pairs: list[Pair] = []

    for position, pair in enumerate(ordered_pairs, start=1):
        if position % val_every == 0:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)

    if not val_pairs:
        raise RuntimeError(
            f"有效样本数只有 {len(ordered_pairs)}，不足以按每 {val_every} "
            "张取 1 张验证集。"
        )

    return train_pairs, val_pairs


def copy_pair(pair: Pair, output_root: Path, split: str) -> tuple[Path, Path]:
    image_output = output_root / "images" / split / pair.logical_rel
    mask_output = (
        output_root / "masks" / split / pair.logical_rel
    ).with_suffix(".png")

    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pair.image_path, image_output)
    shutil.copy2(pair.mask_path, mask_output)

    return image_output, mask_output


def count_where(pairs: Iterable[Pair], *, source_group: str | None = None,
                mask_source: str | None = None) -> int:
    count = 0
    for pair in pairs:
        if source_group is not None and pair.source_group != source_group:
            continue
        if mask_source is not None and pair.mask_source != mask_source:
            continue
        count += 1
    return count


def main() -> int:
    args = parse_args()

    if args.val_every < 2:
        print("[ERROR] --val-every 必须至少为 2", file=sys.stderr)
        return 1

    root = resolve_path(args.root)
    datasets_root = root / "datasets"

    previous_root = resolve_path(
        args.previous_dataset
        if args.previous_dataset is not None
        else datasets_root / "bev_sem_verify"
    )
    seg_pt_root = resolve_path(
        args.seg_pt_root
        if args.seg_pt_root is not None
        else datasets_root / "seg_pt"
    )
    sem_train_bev_root = datasets_root / "bev_all"
    fallback_bev_root = resolve_path(args.bev_root)
    output_root = datasets_root / args.output_name

    print("=" * 92)
    print("Prepare merged dense semantic dataset")
    print(f"Previous dataset : {previous_root}")
    print(f"New seg_pt root  : {seg_pt_root}")
    print(f"sem_train BEV    : {sem_train_bev_root}")
    print(f"Fallback BEV     : {fallback_bev_root}")
    print(f"Output dataset   : {output_root}")
    print(f"Validation rule  : every {args.val_every}th sample")
    print(f"New runs         : {', '.join(args.new_runs)}")
    print("=" * 92)

    try:
        if not previous_root.is_dir():
            raise RuntimeError(f"旧数据集不存在: {previous_root}")
        if not seg_pt_root.is_dir():
            raise RuntimeError(f"seg_pt 目录不存在: {seg_pt_root}")

        # Stage 1: organize and merge all new seg_pt samples.
        new_pairs = load_new_pairs(
            seg_pt_root=seg_pt_root,
            sem_train_bev_root=sem_train_bev_root,
            fallback_bev_root=fallback_bev_root,
            new_run_dirs=args.new_runs,
            corrected_dir_name=args.corrected_dir_name,
            auto_dir_name=args.auto_dir_name,
        )

        # Stage 2: load the old prepared dataset and merge with new data.
        previous_pairs = load_previous_pairs(previous_root)
        merged_pairs, replaced_count = merge_pairs(previous_pairs, new_pairs)

        if len(merged_pairs) < args.val_every:
            raise RuntimeError(
                f"合并后只有 {len(merged_pairs)} 个样本，无法执行每 "
                f"{args.val_every} 张取 1 张作为验证集。"
            )

        print("\nValidating all image/mask pairs...")
        for index, pair in enumerate(merged_pairs, start=1):
            validate_pair(pair)
            if index == 1 or index % 200 == 0 or index == len(merged_pairs):
                print(f"  [{index}/{len(merged_pairs)}] validated")

        merged_pairs.sort(key=pair_sort_key)
        train_pairs, val_pairs = split_every_n(
            ordered_pairs=merged_pairs,
            val_every=args.val_every,
        )

        if output_root.exists():
            if not args.overwrite:
                raise RuntimeError(
                    f"输出目录已存在: {output_root}\n"
                    "请加 --overwrite 重新生成。"
                )
            shutil.rmtree(output_root)

        output_root.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, object]] = []
        global_position = {
            pair.duplicate_key: position
            for position, pair in enumerate(merged_pairs, start=1)
        }

        for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
            for index, pair in enumerate(split_pairs, start=1):
                image_output, mask_output = copy_pair(pair, output_root, split)
                records.append({
                    "global_position": global_position[pair.duplicate_key],
                    "split": split,
                    "run_name": pair.run_name,
                    "logical_path": pair.logical_rel.as_posix(),
                    "source_group": pair.source_group,
                    "mask_source": pair.mask_source,
                    "source_detail": pair.source_detail,
                    "class_ids": ",".join(map(str, pair.class_ids or [])),
                    "width": pair.width,
                    "height": pair.height,
                    "source_image": str(pair.image_path),
                    "source_mask": str(pair.mask_path),
                    "output_image": str(image_output),
                    "output_mask": str(mask_output),
                })
                if index == 1 or index % 200 == 0 or index == len(split_pairs):
                    print(f"[{split}] {index}/{len(split_pairs)} copied")

        records.sort(key=lambda row: int(row["global_position"]))

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
                "global_position",
                "split",
                "run_name",
                "logical_path",
                "source_group",
                "mask_source",
                "source_detail",
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

        print("\n数据集建立完成")
        print("=" * 92)
        print(f"Previous samples          : {len(previous_pairs)}")
        print(f"New samples total         : {len(new_pairs)}")
        print(
            "  corrected masks         : "
            f"{count_where(new_pairs, mask_source='corrected')}"
        )
        print(
            "  autolabel fallback masks: "
            f"{count_where(new_pairs, mask_source='autolabel')}"
        )
        print(f"New samples replacing old : {replaced_count}")
        print(f"Merged unique samples     : {len(merged_pairs)}")
        print("-" * 92)
        print(f"Train samples             : {len(train_pairs)}")
        print(f"Validation samples        : {len(val_pairs)}")
        print(
            "Validation positions      : "
            f"{args.val_every}, {args.val_every * 2}, "
            f"{args.val_every * 3}, ..."
        )
        print("-" * 92)
        print(f"Dataset  : {output_root}")
        print(f"YAML     : {yaml_path}")
        print(f"Manifest : {manifest_path}")
        print("=" * 92)

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
