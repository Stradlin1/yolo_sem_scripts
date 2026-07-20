#!/usr/bin/env python3
import argparse
import random
import shutil
from pathlib import Path


FOLDERS = [
    "2026-06-28_01-05-54",
    "2026-06-28_01-08-45",
    "2026-06-28_01-13-14",
    "2026-06-28_01-18-35",
    "2026-06-28_01-25-30",
    "2026-06-28_01-30-09",
]


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]


def find_image(bev_dir, stem):
    for ext in IMAGE_EXTS:
        p = bev_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def copy_pair(img_path, label_path, out_img_dir, out_label_dir, new_stem):
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    out_img_path = out_img_dir / f"{new_stem}{img_path.suffix.lower()}"
    out_label_path = out_label_dir / f"{new_stem}.txt"

    shutil.copy2(str(img_path), str(out_img_path))
    shutil.copy2(str(label_path), str(out_label_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bev-root",
        required=True,
        help="Root of original BEV folders, e.g. data/extracted/bev_vehicle",
    )
    parser.add_argument(
        "--fixed-root",
        required=True,
        help="Root of fixed labels, e.g. data/fixed_bev_labels_v2",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output YOLO dataset dir.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--test-folder",
        default="",
        help="Optional: reserve one whole sequence folder as test set.",
    )
    args = parser.parse_args()

    bev_root = Path(args.bev_root)
    fixed_root = Path(args.fixed_root)
    out_root = Path(args.out)

    random.seed(args.seed)

    all_items = []
    test_items = []

    for folder in FOLDERS:
        bev_dir = bev_root / folder / "bev"
        label_dir = fixed_root / folder / "labels"

        if not bev_dir.exists():
            print(f"[WARN] missing bev dir: {bev_dir}")
            continue

        if not label_dir.exists():
            print(f"[WARN] missing label dir: {label_dir}")
            continue

        label_paths = sorted(label_dir.glob("*.txt"))

        for label_path in label_paths:
            stem = label_path.stem
            img_path = find_image(bev_dir, stem)

            if img_path is None:
                print(f"[WARN] image not found for label: {label_path}")
                continue

            item = {
                "folder": folder,
                "img": img_path,
                "label": label_path,
                "stem": stem,
            }

            if args.test_folder and folder == args.test_folder:
                test_items.append(item)
            else:
                all_items.append(item)

    if not all_items and not test_items:
        raise RuntimeError("No image-label pairs found.")

    random.shuffle(all_items)

    if args.test_folder:
        train_val_items = all_items
        n = len(train_val_items)
        n_train = int(n * args.train_ratio / (args.train_ratio + args.val_ratio))
        train_items = train_val_items[:n_train]
        val_items = train_val_items[n_train:]
    else:
        n = len(all_items)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        train_items = all_items[:n_train]
        val_items = all_items[n_train:n_train + n_val]
        test_items = all_items[n_train + n_val:]

    splits = {
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }

    for split, items in splits.items():
        for item in items:
            new_stem = f"{item['folder']}__{item['stem']}"
            copy_pair(
                img_path=item["img"],
                label_path=item["label"],
                out_img_dir=out_root / "images" / split,
                out_label_dir=out_root / "labels" / split,
                new_stem=new_stem,
            )

    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {out_root}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write("names:\n")
        f.write("  0: P\n")
        f.write("  1: A_DaTing\n")
        f.write("  2: B_TongDao\n")
        f.write("  3: C_ZhenLiaoShi\n")
        f.write("  4: JinQu\n")

    print("=" * 80)
    print("Dataset built.")
    print("Output:", out_root)
    print("train:", len(train_items))
    print("val  :", len(val_items))
    print("test :", len(test_items))
    print("data :", data_yaml)
    print("=" * 80)


if __name__ == "__main__":
    main()