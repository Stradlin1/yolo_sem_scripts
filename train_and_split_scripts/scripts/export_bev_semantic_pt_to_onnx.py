#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnxruntime as ort
from ultralytics import YOLO


EXPECTED_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the BEV dense semantic PT checkpoint to ONNX."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/home/xhm/Desktop/sem_train/"
            "runs/round2_merged_071801_finetune/weights/best.pt"
        ),
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


def normalize_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(raw_names)
        }
    raise TypeError(f"Unsupported names type: {type(raw_names).__name__}")


def inspect_onnx(onnx_path: Path) -> None:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    print("\nONNX Runtime inspection")
    print("=" * 88)
    print("Inputs:")
    for item in session.get_inputs():
        print(f"  name={item.name}, shape={item.shape}, type={item.type}")

    print("Outputs:")
    for item in session.get_outputs():
        print(f"  name={item.name}, shape={item.shape}, type={item.type}")

    metadata = session.get_modelmeta().custom_metadata_map
    print("Metadata:")
    if metadata:
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("  <empty>")
    print("=" * 88)


def main() -> int:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"[ERROR] PT checkpoint not found: {model_path}", file=sys.stderr)
        return 1

    print("=" * 88)
    print("Dense semantic PT -> ONNX")
    print(f"PT model  : {model_path}")
    print(f"Input     : {args.batch} x 3 x {args.imgsz} x {args.imgsz}")
    print(f"Dynamic   : {args.dynamic}")
    print(f"Simplify  : {args.simplify}")
    print(f"Opset     : {args.opset}")
    print(f"Device    : {args.device}")
    print("=" * 88)

    model = YOLO(str(model_path), task="semantic")

    checkpoint_names = normalize_names(model.names)
    print("Checkpoint names:")
    for class_id, class_name in checkpoint_names.items():
        print(f"  {class_id}: {class_name}")

    if checkpoint_names != EXPECTED_NAMES:
        print("\n[WARN] Correcting class-name metadata before export.")
        print("       Numeric class IDs and output-channel order are unchanged.")
        model.model.names = EXPECTED_NAMES.copy()

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        device=args.device,
    )

    onnx_path = Path(str(exported)).expanduser().resolve()
    if not onnx_path.is_file():
        print(
            f"[ERROR] Export returned a path that does not exist: {onnx_path}",
            file=sys.stderr,
        )
        return 2

    print(f"\nExport complete: {onnx_path}")
    inspect_onnx(onnx_path)

    print(
        "\nInspect the output shape before writing post-processing.\n"
        "Possible dense-semantic layouts:\n"
        "  [1, 6, H, W] -> use argmax(axis=1)\n"
        "  [1, H, W]    -> direct class-ID map\n"
        "Background class ID is 5, not 0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
