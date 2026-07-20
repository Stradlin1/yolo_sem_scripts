#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export YOLO26 dense semantic-segmentation PT checkpoint to ONNX opset 11.

Default input:
  /home/xhm/Desktop/sem_train/runs/round2_merged_071801_finetune/weights/best.pt

Default output:
  best.onnx in the same directory as best.pt

Class mapping:
  0 JinQu
  1 C_ZhenLiaoShi
  2 B_TongDao
  3 A_DaTing
  4 P
  5 background

The script only corrects class-name metadata when necessary.
It does not reorder output channels or change numeric class IDs.
"""

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

DEFAULT_MODEL = Path(
    "/home/xhm/Desktop/sem_train/"
    "runs/round2_merged_071801_finetune/weights/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export BEV semantic PT model to ONNX with opset 11."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Input PT checkpoint. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Fixed square input size. Default: 320.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Fixed batch size. Default: 1.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Default: cpu.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX graph after export. Disabled by default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing .onnx file.",
    )
    return parser.parse_args()


def normalize_names(raw_names: object) -> dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(key): str(value) for key, value in raw_names.items()}

    if isinstance(raw_names, (list, tuple)):
        return {
            class_id: str(class_name)
            for class_id, class_name in enumerate(raw_names)
        }

    raise TypeError(
        f"Unsupported model.names type: {type(raw_names).__name__}"
    )


def inspect_onnx(onnx_path: Path) -> None:
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    print("\nONNX Runtime inspection")
    print("=" * 88)

    print("Inputs:")
    for item in session.get_inputs():
        print(
            f"  name={item.name}, "
            f"shape={item.shape}, "
            f"type={item.type}"
        )

    print("Outputs:")
    for item in session.get_outputs():
        print(
            f"  name={item.name}, "
            f"shape={item.shape}, "
            f"type={item.type}"
        )

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
    output_path = model_path.with_suffix(".onnx")

    if not model_path.is_file():
        print(
            f"[ERROR] PT checkpoint not found: {model_path}",
            file=sys.stderr,
        )
        return 1

    if args.imgsz <= 0:
        print("[ERROR] --imgsz must be greater than 0", file=sys.stderr)
        return 1

    if args.batch <= 0:
        print("[ERROR] --batch must be greater than 0", file=sys.stderr)
        return 1

    if output_path.exists() and not args.overwrite:
        print(
            f"[ERROR] ONNX file already exists: {output_path}\n"
            "Use --overwrite to replace it.",
            file=sys.stderr,
        )
        return 2

    if output_path.exists() and args.overwrite:
        output_path.unlink()

    print("=" * 88)
    print("Dense semantic PT -> ONNX")
    print(f"PT model  : {model_path}")
    print(f"ONNX path : {output_path}")
    print(f"Input     : {args.batch} x 3 x {args.imgsz} x {args.imgsz}")
    print("Dynamic   : False")
    print(f"Simplify  : {args.simplify}")
    print("Opset     : 11")
    print(f"Device    : {args.device}")
    print("=" * 88)

    model = YOLO(str(model_path), task="semantic")

    checkpoint_names = normalize_names(model.names)

    print("Checkpoint names:")
    for class_id, class_name in checkpoint_names.items():
        print(f"  {class_id}: {class_name}")

    if checkpoint_names != EXPECTED_NAMES:
        print("\n[WARN] Correcting class-name metadata before export.")
        print("       Numeric class IDs and output channels are unchanged.")
        print(f"       checkpoint: {checkpoint_names}")
        print(f"       corrected : {EXPECTED_NAMES}")
        model.model.names = EXPECTED_NAMES.copy()

    exported_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=False,
        simplify=args.simplify,
        opset=11,
        device=args.device,
    )

    exported_path = Path(str(exported_path)).expanduser().resolve()

    if not exported_path.is_file():
        print(
            f"[ERROR] Export returned a missing file: {exported_path}",
            file=sys.stderr,
        )
        return 3

    if exported_path != output_path:
        print(
            f"[WARN] Exported path differs from expected path:\n"
            f"       expected: {output_path}\n"
            f"       actual  : {exported_path}"
        )

    print(f"\nExport complete: {exported_path}")

    inspect_onnx(exported_path)

    print("\nExpected deployment interface:")
    print("  input : float32 [1, 3, 320, 320]")
    print("  output: uint8   [1, 320, 320]")
    print("  output is already the class-ID mask; do not run argmax.")
    print("  background class ID is 5.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
