#!/usr/bin/env python3
from pathlib import Path
from ultralytics import YOLO
import ultralytics


def main():
    # =========================
    # Paths
    # =========================
    workspace = Path.home() / "Desktop/relocate_ws"

    data_yaml = workspace / "datasets/bev_seg_fixed_v2/data.yaml"
    model_path = workspace / "models/yolo11n-seg.pt"

    project_dir = workspace / "runs/bev_seg"
    run_name = "yolo11n_fixed_v2_local"

    # =========================
    # Safety checks
    # =========================
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model not found: {model_path}\n"
            f"Put yolo11n-seg.pt here first:\n"
            f"  {model_path}"
        )

    if model_path.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"Model file is too small and may be invalid: {model_path}\n"
            f"Size: {model_path.stat().st_size} bytes"
        )

    print("=" * 80)
    print("Ultralytics version:", ultralytics.__version__)
    print("Using local model  :", model_path)
    print("Using data yaml    :", data_yaml)
    print("Project dir        :", project_dir)
    print("Run name           :", run_name)
    print("=" * 80)

    # =========================
    # Model
    # =========================
    # 必须用本地绝对路径，不写 YOLO('yolo11n-seg.pt')
    model = YOLO(str(model_path))

    print("Loaded model task:", model.task)

    if model.task != "segment":
        raise RuntimeError(
            f"Loaded model task is not segment: {model.task}\n"
            f"Check whether this file is really a segmentation model:\n"
            f"  {model_path}"
        )

    # =========================
    # Train
    # =========================
    results = model.train(
        data=str(data_yaml),
        imgsz=320,
        epochs=800,
        batch=16,
        device=0,

        project=str(project_dir),
        name=run_name,
        exist_ok=True,

        # segmentation task
        task="segment",
        amp=False,

        # training settings
        workers=4,
        patience=80,
        save=True,
        save_period=10,
        cache=False,

        # optimizer settings
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,

        # augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.3,
        mixup=0.0,
        copy_paste=0.0,

        verbose=True,
    )

    best_pt = project_dir / run_name / "weights/best.pt"
    last_pt = project_dir / run_name / "weights/last.pt"

    print("=" * 80)
    print("Training finished.")
    print("Run dir:")
    print(project_dir / run_name)
    print("Best model:")
    print(best_pt)
    print("Last model:")
    print(last_pt)
    print("=" * 80)


if __name__ == "__main__":
    main()