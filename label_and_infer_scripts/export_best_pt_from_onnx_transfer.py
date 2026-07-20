#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


DEFAULT_PT = Path(
    "/home/xhm/Desktop/onnx_transfer/best.pt"
)

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
        description="Export the trained BEV semantic model to ONNX."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_PT,
        help=f"Input best.pt. Default: {DEFAULT_PT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional final ONNX path. "
            "If omitted, the ONNX remains beside best.pt."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Default: cpu",
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
            index: str(class_name)
            for index, class_name in enumerate(raw_names)
        }

    raise TypeError(
        f"Unsupported model.names type: {type(raw_names).__name__}"
    )


def inspect_onnx(onnx_path: Path) -> None:
    try:
        import onnx
    except ImportError:
        print("[WARN] onnx is not installed; graph validation skipped.")
        return

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    print("\nONNX graph check passed")
    print("-" * 88)

    initializer_names = {
        initializer.name for initializer in model.graph.initializer
    }
    graph_inputs = [
        tensor
        for tensor in model.graph.input
        if tensor.name not in initializer_names
    ]

    print("Inputs:")
    for tensor in graph_inputs:
        dims = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(dim.dim_value)
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append("?")
        print(f"  {tensor.name}: {dims}")

    print("Outputs:")
    for tensor in model.graph.output:
        dims = []
        for dim in tensor.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(dim.dim_value)
            elif dim.dim_param:
                dims.append(dim.dim_param)
            else:
                dims.append("?")

        elem_type = tensor.type.tensor_type.elem_type
        print(f"  {tensor.name}: shape={dims}, elem_type={elem_type}")


def main() -> int:
    args = parse_args()

    pt_path = args.model.expanduser().resolve()

    if not pt_path.is_file():
        print(f"[ERROR] Model not found: {pt_path}", file=sys.stderr)
        return 1

    print("=" * 88)
    print("BEV semantic PT -> ONNX export")
    print(f"Input PT : {pt_path}")
    print("Image    : 320 x 320")
    print("Batch    : 1")
    print("Opset    : 11")
    print("Dynamic  : False")
    print("Simplify : False")
    print(f"Device   : {args.device}")
    print("=" * 88)

    model = YOLO(str(pt_path), task="semantic")

    actual_names = normalize_names(model.names)
    if actual_names != EXPECTED_NAMES:
        print(
            "[ERROR] Model class mapping does not match the project protocol.",
            file=sys.stderr,
        )
        print(f"Actual  : {actual_names}", file=sys.stderr)
        print(f"Expected: {EXPECTED_NAMES}", file=sys.stderr)
        return 2

    export_result = model.export(
        format="onnx",
        imgsz=320,
        batch=1,
        opset=11,
        dynamic=False,
        simplify=False,
        device=args.device,
    )

    exported_path = Path(str(export_result)).expanduser().resolve()

    if not exported_path.is_file():
        fallback = pt_path.with_suffix(".onnx")
        if fallback.is_file():
            exported_path = fallback
        else:
            print(
                "[ERROR] Export completed but ONNX file was not found.",
                file=sys.stderr,
            )
            print(f"Export result: {export_result}", file=sys.stderr)
            return 3

    final_path = exported_path

    if args.output is not None:
        requested_output = args.output.expanduser().resolve()
        requested_output.parent.mkdir(parents=True, exist_ok=True)

        if requested_output != exported_path:
            shutil.copy2(exported_path, requested_output)

        final_path = requested_output

    inspect_onnx(final_path)

    print("\nExport completed")
    print("=" * 88)
    print(f"ONNX: {final_path}")
    print("Class mapping:")
    for class_id, class_name in EXPECTED_NAMES.items():
        print(f"  {class_id}: {class_name}")
    print("=" * 88)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
