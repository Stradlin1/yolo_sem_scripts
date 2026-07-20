#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train YOLO26n semantic segmentation on the BEV verification dataset.

This script deliberately requires a LOCAL pretrained weight file so that
Ultralytics will not try to download weights during training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

WORKSPACE = Path("/home/xhm/Desktop/sem_train")

MODEL_PATH = WORKSPACE / "models" / "yolo26n-sem.pt"
DATA_YAML = WORKSPACE / "datasets" / "bev_sem_verify" / "data.yaml"

PROJECT_DIR = WORKSPACE / "runs"
RUN_NAME = "verify_2026-06-28_01-08-45"


# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------

IMAGE_SIZE = 320
EPOCHS = 300
BATCH_SIZE = 8
DEVICE = 0
WORKERS = 8
SEED = 42

# The current train/val split comes from one recording and is intended only
# to verify the training pipeline. Conservative augmentation is used here.
TRAIN_ARGS = {
    "data": str(DATA_YAML),
    "imgsz": IMAGE_SIZE,
    "epochs": EPOCHS,
    "batch": BATCH_SIZE,
    "device": DEVICE,
    "workers": WORKERS,
    "project": str(PROJECT_DIR),
    "name": RUN_NAME,
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "auto",
    "amp": True,
    "seed": SEED,
    "deterministic": True,
    "patience": 50,
    "save": True,
    "save_period": 5,
    "plots": True,
    "verbose": True,

    # Avoid geometric and color augmentation during this first verification
    # run. BEV orientation and floor appearance have physical meaning.
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


def check_environment() -> None:
    print("=" * 80)
    print("YOLO semantic training")
    print(f"Ultralytics : {ultralytics.__version__}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU         : {torch.cuda.get_device_name(DEVICE)}")

    print(f"Model       : {MODEL_PATH}")
    print(f"Data YAML   : {DATA_YAML}")
    print(f"Output      : {PROJECT_DIR / RUN_NAME}")
    print("=" * 80)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "\nLocal pretrained weights were not found:\n"
            f"  {MODEL_PATH}\n\n"
            "Download yolo26n-sem.pt first and place it in the models directory."
        )

    if not DATA_YAML.is_file():
        raise FileNotFoundError(
            "\nDataset YAML was not found:\n"
            f"  {DATA_YAML}\n\n"
            "Run prepare_verify_dataset.py before training."
        )

    if DEVICE != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "DEVICE requests CUDA, but torch.cuda.is_available() is False."
        )

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    try:
        check_environment()

        # Passing an absolute local path prevents automatic weight downloading.
        model = YOLO(str(MODEL_PATH), task="semantic")

        print(f"Detected task: {model.task}")
        print("Starting training...")

        results = model.train(**TRAIN_ARGS)

        output_dir = PROJECT_DIR / RUN_NAME
        best_path = output_dir / "weights" / "best.pt"
        last_path = output_dir / "weights" / "last.pt"

        print("=" * 80)
        print("Training finished")
        print(f"Run directory : {output_dir}")
        print(f"Best weights  : {best_path}")
        print(f"Last weights  : {last_path}")
        print("=" * 80)

        # Keep a reference so linters do not mark the training result unused.
        _ = results
        return 0

    except KeyboardInterrupt:
        print("\n[STOPPED] Training interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
