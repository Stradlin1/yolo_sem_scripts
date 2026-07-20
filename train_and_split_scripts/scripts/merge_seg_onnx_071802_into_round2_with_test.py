#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 seg_onnx_071802 中的新标注数据合并进现有 bev_sem_round2_merged。

默认目录结构：

/home/xhm/Desktop/sem_train/datasets/
├── bev_all/
│   ├── 2026-06-28_01-13-14/
│   ├── 2026-06-28_01-18-35/
│   ├── 2026-06-28_01-25-30/
│   └── 2026-06-28_01-30-09/
├── seg_onnx_071802/
│   ├── 2026-06-28_01-13-14/
│   │   ├── masks_id/
│   │   └── masks_corrected/
│   └── ...
└── bev_sem_round2_merged/
    ├── images/train
    ├── images/val
    ├── masks/train
    ├── masks/val
    ├── data.yaml
    └── manifest.csv

处理规则：

1. 对每个新 run：
   - 如果某张图片存在 masks_corrected/xxx.png，使用人工修正标签；
   - 否则使用 masks_id/xxx.png；
   - 两者都没有则停止并报错。

2. 训练/验证数据：
   - 2026-06-28_01-13-14
   - 2026-06-28_01-18-35
   - 2026-06-28_01-25-30
   这三批会与现有 bev_sem_round2_merged 合并。
   其中旧的 2026-06-28_01-13-14 会被整批替换。

3. 测试数据：
   - 2026-06-28_01-30-09
   只进入 images/test 和 masks/test，不进入 train 或 val。

4. train/val 对全部非测试数据按时间和文件名排序后划分：
   每 10 张中的第 10 张进入 val，其余进入 train。

5. 先在临时目录中完整构建并校验，成功后再替换正式目录。
   原 bev_sem_round2_merged 会自动改名为带时间戳的备份目录。

类别映射：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background
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

DEFAULT_TRAIN_RUNS = [
    "2026-06-28_01-13-14",
    "2026-06-28_01-18-35",
    "2026-06-28_01-25-30",
]

DEFAULT_TEST_RUNS = [
    "2026-06-28_01-30-09",
]

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
        return self.logical_rel.with_suffix("").as_posix().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge corrected/automatic ONNX semantic labels into the existing "
            "bev_sem_round2_merged dataset."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/xhm/Desktop/sem_train"),
        help="sem_train 根目录。",
    )
    parser.add_argument(
        "--dataset-name",
        default="bev_sem_round2_merged",
        help="要更新的数据集目录名。",
    )
    parser.add_argument(
        "--bev-root",
        type=Path,
        default=None,
        help="默认：<root>/datasets/bev_all",
    )
    parser.add_argument(
        "--seg-root",
        type=Path,
        default=None,
        help="默认：<root>/datasets/seg_onnx_071802",
    )
    parser.add_argument(
        "--train-runs",
        nargs="+",
        default=DEFAULT_TRAIN_RUNS,
        help="本次要合并到 train/val 的 run。",
    )
    parser.add_argument(
        "--test-runs",
        nargs="+",
        default=DEFAULT_TEST_RUNS,
        help="只放入 test，不进入 train/val 的 run。",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=10,
        help="每 N 张中的第 N 张放入验证集，默认 10。",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="成功后不保留旧数据集备份。",
    )
    return parser.parse_args()


def natural_tokens(text: str) -> tuple[object, ...]:
    parts = NATURAL_PATTERN.split(text.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def extract_run_name(value: str | Path) -> str:
    match = RUN_PATTERN.search(str(value))
    return match.group(0) if match else ""


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


def load_manifest(dataset_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    manifest_path = dataset_root / "manifest.csv"
    rows: dict[tuple[str, str], dict[str, str]] = {}

    if not manifest_path.is_file():
        return rows

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            split = (row.get("split") or "").strip()
            relative = (
                row.get("logical_path")
                or row.get("relative_path")
                or ""
            ).strip()

            if split and relative:
                rows[(split, Path(relative).as_posix())] = row

    return rows


def load_existing_pairs(dataset_root: Path) -> list[Pair]:
    """
    读取现有 merged 数据集，忽略原 train/val 划分。
    后面会对全部数据重新按每 10 张第 10 张划分。
    """
    manifest = load_manifest(dataset_root)
    pairs: list[Pair] = []

    for old_split in ("train", "val"):
        image_root = dataset_root / "images" / old_split
        mask_root = dataset_root / "masks" / old_split

        if not image_root.is_dir():
            raise RuntimeError(f"现有数据集图片目录不存在: {image_root}")
        if not mask_root.is_dir():
            raise RuntimeError(f"现有数据集标签目录不存在: {mask_root}")

        for image_path in find_images(image_root):
            relative_path = image_path.relative_to(image_root)
            mask_path = (mask_root / relative_path).with_suffix(".png")

            if not mask_path.is_file():
                raise RuntimeError(
                    "现有数据集图片缺少对应 mask:\n"
                    f"  image: {image_path}\n"
                    f"  mask : {mask_path}"
                )

            row = manifest.get((old_split, relative_path.as_posix()), {})

            run_name = (
                (row.get("run_name") or "").strip()
                or extract_run_name(row.get("source_image", ""))
                or extract_run_name(relative_path)
                or extract_run_name(image_path)
            )

            logical_text = (
                row.get("logical_path")
                or row.get("relative_path")
                or ""
            ).strip()

            if logical_text:
                logical_rel = Path(logical_text)
            else:
                tail = path_after_run(relative_path, run_name)
                if run_name and tail is not None:
                    logical_rel = Path(run_name) / tail
                elif run_name:
                    logical_rel = Path(run_name) / relative_path.name
                else:
                    logical_rel = Path("previous_unknown") / relative_path

            pairs.append(Pair(
                image_path=image_path,
                mask_path=mask_path,
                logical_rel=logical_rel,
                run_name=run_name,
                source_group="existing_merged",
                mask_source=(
                    row.get("mask_source", "").strip()
                    or "existing_dataset"
                ),
                source_detail=(
                    row.get("source_detail", "").strip()
                    or f"{dataset_root.name}/{old_split}"
                ),
            ))

    return pairs


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
        return auto_mask, "onnx_autolabel"

    raise RuntimeError(
        "新数据图片同时缺少人工标签和自动标签:\n"
        f"  corrected: {corrected_mask}\n"
        f"  automatic: {auto_mask}"
    )


def load_new_pairs(
    bev_root: Path,
    seg_root: Path,
    run_names: list[str],
) -> list[Pair]:
    """
    先在每个 run 内合并：
      masks_corrected 优先；
      缺失时退回 masks_id。
    """
    pairs: list[Pair] = []

    for run_name in run_names:
        image_root = bev_root / run_name
        run_root = seg_root / run_name
        corrected_root = run_root / "masks_corrected"
        auto_root = run_root / "masks_id"

        if not image_root.is_dir():
            raise RuntimeError(f"BEV 图片目录不存在: {image_root}")
        if not run_root.is_dir():
            raise RuntimeError(f"ONNX 标签 run 不存在: {run_root}")
        if not auto_root.is_dir():
            raise RuntimeError(f"ONNX 自动标签目录不存在: {auto_root}")

        images = find_images(image_root)
        if not images:
            raise RuntimeError(f"BEV 图片目录中没有图片: {image_root}")

        corrected_count = 0
        auto_count = 0

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
                auto_count += 1

            pairs.append(Pair(
                image_path=image_path,
                mask_path=mask_path,
                logical_rel=Path(run_name) / relative_path,
                run_name=run_name,
                source_group="seg_onnx_071802",
                mask_source=mask_source,
                source_detail=run_name,
            ))

        print(
            f"[NEW] {run_name}: total={len(images)}, "
            f"corrected={corrected_count}, "
            f"onnx_fallback={auto_count}"
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
            f"mask 必须是整数类别值: {pair.mask_path}, dtype={mask.dtype}"
        )

    image_h, image_w = image.shape[:2]
    mask_h, mask_w = mask.shape[:2]

    if (image_h, image_w) != (mask_h, mask_w):
        raise RuntimeError(
            "图片和 mask 尺寸不一致:\n"
            f"  image: {pair.image_path} -> {image_w}x{image_h}\n"
            f"  mask : {pair.mask_path} -> {mask_w}x{mask_h}"
        )

    class_ids = {int(value) for value in np.unique(mask)}
    invalid_ids = sorted(class_ids - VALID_CLASS_IDS)

    if invalid_ids:
        raise RuntimeError(
            f"mask 存在非法类别值: {pair.mask_path}, "
            f"invalid={invalid_ids}"
        )

    pair.class_ids = sorted(class_ids)
    pair.width = image_w
    pair.height = image_h


def merge_replacing_runs(
    existing_pairs: list[Pair],
    new_pairs: list[Pair],
    replaced_runs: set[str],
) -> tuple[list[Pair], int]:
    """
    先删除现有 merged 中本次 run 的全部旧样本，再加入新样本。

    这比仅按同名图片覆盖更严格，可防止旧的 01-13-14 样本残留。
    """
    retained: dict[str, Pair] = {}
    removed_count = 0

    for pair in existing_pairs:
        if pair.run_name in replaced_runs:
            removed_count += 1
            continue

        key = pair.duplicate_key
        if key in retained:
            raise RuntimeError(
                "现有 merged 数据集内部存在重复样本:\n"
                f"  key   : {key}\n"
                f"  first : {retained[key].image_path}\n"
                f"  second: {pair.image_path}"
            )

        retained[key] = pair

    for pair in new_pairs:
        key = pair.duplicate_key

        if key in retained:
            raise RuntimeError(
                "新数据和保留的旧数据发生非预期重复:\n"
                f"  key: {key}\n"
                f"  old: {retained[key].image_path}\n"
                f"  new: {pair.image_path}"
            )

        retained[key] = pair

    return list(retained.values()), removed_count


def split_every_n(
    ordered_pairs: list[Pair],
    val_every: int,
) -> tuple[list[Pair], list[Pair]]:
    train_pairs: list[Pair] = []
    val_pairs: list[Pair] = []

    for position, pair in enumerate(ordered_pairs, start=1):
        if position % val_every == 0:
            val_pairs.append(pair)
        else:
            train_pairs.append(pair)

    if not val_pairs:
        raise RuntimeError(
            f"总样本数 {len(ordered_pairs)} 不足以按每 {val_every} "
            "张取 1 张作为验证集。"
        )

    return train_pairs, val_pairs


def copy_pair(
    pair: Pair,
    build_root: Path,
    split: str,
) -> tuple[Path, Path]:
    image_output = build_root / "images" / split / pair.logical_rel
    mask_output = (
        build_root / "masks" / split / pair.logical_rel
    ).with_suffix(".png")

    image_output.parent.mkdir(parents=True, exist_ok=True)
    mask_output.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(pair.image_path, image_output)
    shutil.copy2(pair.mask_path, mask_output)

    return image_output, mask_output


def count_masks(pairs: Iterable[Pair], source: str) -> int:
    return sum(pair.mask_source == source for pair in pairs)


def verify_built_dataset(
    build_root: Path,
    expected_train: int,
    expected_val: int,
    expected_test: int,
) -> None:
    for split, expected in (
        ("train", expected_train),
        ("val", expected_val),
        ("test", expected_test),
    ):
        image_count = len(find_images(build_root / "images" / split))
        mask_count = len(list((build_root / "masks" / split).rglob("*.png")))

        if image_count != expected:
            raise RuntimeError(
                f"构建后 {split} 图片数量错误: "
                f"expected={expected}, actual={image_count}"
            )

        if mask_count != expected:
            raise RuntimeError(
                f"构建后 {split} mask 数量错误: "
                f"expected={expected}, actual={mask_count}"
            )


def main() -> int:
    args = parse_args()

    if args.val_every < 2:
        print("[ERROR] --val-every 必须至少为 2", file=sys.stderr)
        return 1

    root = args.root.expanduser().resolve()
    datasets_root = root / "datasets"

    dataset_root = datasets_root / args.dataset_name
    bev_root = (
        args.bev_root.expanduser().resolve()
        if args.bev_root is not None
        else datasets_root / "bev_all"
    )
    seg_root = (
        args.seg_root.expanduser().resolve()
        if args.seg_root is not None
        else datasets_root / "seg_onnx_071802"
    )

    build_root = datasets_root / f".{args.dataset_name}_building"

    print("=" * 96)
    print("Merge seg_onnx_071802 into existing semantic dataset")
    print(f"Existing dataset : {dataset_root}")
    print(f"BEV root         : {bev_root}")
    print(f"ONNX label root  : {seg_root}")
    print(f"Build directory  : {build_root}")
    print(f"Train/val runs   : {', '.join(args.train_runs)}")
    print(f"Test-only runs   : {', '.join(args.test_runs)}")
    print(f"Validation rule  : every {args.val_every}th non-test sample")
    print("=" * 96)

    try:
        if not dataset_root.is_dir():
            raise RuntimeError(f"现有 merged 数据集不存在: {dataset_root}")
        if not bev_root.is_dir():
            raise RuntimeError(f"bev_all 不存在: {bev_root}")
        if not seg_root.is_dir():
            raise RuntimeError(f"seg_onnx_071802 不存在: {seg_root}")

        overlap = set(args.train_runs) & set(args.test_runs)
        if overlap:
            raise RuntimeError(
                f"同一个 run 不能同时属于 train/val 和 test: {sorted(overlap)}"
            )

        # 第一步：分别整理训练候选 run 和测试 run。
        train_new_pairs = load_new_pairs(
            bev_root=bev_root,
            seg_root=seg_root,
            run_names=args.train_runs,
        )
        test_pairs = load_new_pairs(
            bev_root=bev_root,
            seg_root=seg_root,
            run_names=args.test_runs,
        )

        # 第二步：读取现有 merged。
        # train-runs 和 test-runs 的旧版本全部删除，避免 01-13 或 01-30 残留。
        existing_pairs = load_existing_pairs(dataset_root)
        replaced_runs = set(args.train_runs) | set(args.test_runs)

        merged_pairs, removed_old_count = merge_replacing_runs(
            existing_pairs=existing_pairs,
            new_pairs=train_new_pairs,
            replaced_runs=replaced_runs,
        )

        if len(merged_pairs) < args.val_every:
            raise RuntimeError(
                f"合并后非测试样本只有 {len(merged_pairs)} 个，无法划分验证集。"
            )

        all_pairs_for_validation = merged_pairs + test_pairs

        print("\nValidating image/mask pairs...")
        for index, pair in enumerate(all_pairs_for_validation, start=1):
            validate_pair(pair)

            if (
                index == 1
                or index % 200 == 0
                or index == len(all_pairs_for_validation)
            ):
                print(
                    f"  [{index}/{len(all_pairs_for_validation)}] validated"
                )

        merged_pairs.sort(key=pair_sort_key)
        test_pairs.sort(key=pair_sort_key)

        train_pairs, val_pairs = split_every_n(
            ordered_pairs=merged_pairs,
            val_every=args.val_every,
        )

        if build_root.exists():
            shutil.rmtree(build_root)

        build_root.mkdir(parents=True, exist_ok=True)

        global_position = {
            pair.duplicate_key: position
            for position, pair in enumerate(merged_pairs, start=1)
        }
        test_position = {
            pair.duplicate_key: position
            for position, pair in enumerate(test_pairs, start=1)
        }

        records: list[dict[str, object]] = []

        for split, split_pairs in (
            ("train", train_pairs),
            ("val", val_pairs),
            ("test", test_pairs),
        ):
            for index, pair in enumerate(split_pairs, start=1):
                image_output, mask_output = copy_pair(
                    pair=pair,
                    build_root=build_root,
                    split=split,
                )

                if split == "test":
                    position_value = f"test-{test_position[pair.duplicate_key]}"
                else:
                    position_value = str(global_position[pair.duplicate_key])

                records.append({
                    "global_position": position_value,
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

        def record_sort_key(row: dict[str, object]) -> tuple[int, object]:
            position = str(row["global_position"])
            if position.startswith("test-"):
                return (1, int(position.split("-", 1)[1]))
            return (0, int(position))

        records.sort(key=record_sort_key)

        data_yaml = {
            "path": str(dataset_root),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "masks_dir": "masks",
            "names": CLASS_NAMES,
        }

        yaml_path = build_root / "data.yaml"
        with yaml_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(
                data_yaml,
                file,
                allow_unicode=True,
                sort_keys=False,
            )

        manifest_path = build_root / "manifest.csv"
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

        verify_built_dataset(
            build_root=build_root,
            expected_train=len(train_pairs),
            expected_val=len(val_pairs),
            expected_test=len(test_pairs),
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = datasets_root / (
            f"{args.dataset_name}_backup_{timestamp}"
        )

        dataset_root.rename(backup_root)
        build_root.rename(dataset_root)

        if args.no_backup:
            shutil.rmtree(backup_root)
            backup_message = "<removed by --no-backup>"
        else:
            backup_message = str(backup_root)

        print("\n合并完成")
        print("=" * 96)
        print(f"Existing samples before   : {len(existing_pairs)}")
        print(f"Old samples removed       : {removed_old_count}")
        print(f"Train/val samples added   : {len(train_new_pairs)}")
        print(
            f"  corrected               : "
            f"{count_masks(train_new_pairs, 'corrected')}"
        )
        print(
            f"  ONNX fallback           : "
            f"{count_masks(train_new_pairs, 'onnx_autolabel')}"
        )
        print(f"Test-only samples         : {len(test_pairs)}")
        print(
            f"  corrected               : "
            f"{count_masks(test_pairs, 'corrected')}"
        )
        print(
            f"  ONNX fallback           : "
            f"{count_masks(test_pairs, 'onnx_autolabel')}"
        )
        print(f"Merged non-test samples   : {len(merged_pairs)}")
        print("-" * 96)
        print(f"Train                     : {len(train_pairs)}")
        print(f"Val                       : {len(val_pairs)}")
        print(f"Test                      : {len(test_pairs)}")
        print(
            f"Val positions             : "
            f"{args.val_every}, {args.val_every * 2}, "
            f"{args.val_every * 3}, ..."
        )
        print("-" * 96)
        print(f"Updated dataset           : {dataset_root}")
        print(f"Old dataset backup        : {backup_message}")
        print(f"YAML                      : {dataset_root / 'data.yaml'}")
        print(f"Manifest                  : {dataset_root / 'manifest.csv'}")
        print("=" * 96)

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)

        if build_root.exists():
            shutil.rmtree(build_root)

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
