#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在现有语义分割 mask 上追加 class 6 = Bucket。

目录约定：
    <dataset>/images/<split>/<run>/...
    <dataset>/masks/<split>/<run>/...

输出：
    <dataset>/masks7/<split>/<run>/...
    <dataset>/overlays7/<split>/<run>/...

特点：
1. 原 masks 永远不修改。
2. 启动时为所有图片建立 masks7 和 overlays7；即使某张图没有 Bucket，
   也会保存一份完整的新 mask 和 overlay。
3. 如果 masks7 已存在，默认从 masks7 继续编辑，支持断点续标。
4. 切换图片或退出时自动保存，不依赖按 S。
5. 所有相对路径均相对于脚本所在目录解析。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent

CLASS_NAMES: Dict[int, str] = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
    6: "Bucket",
}

# OpenCV 使用 BGR。颜色只用于显示，不影响训练标签。
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 255),    # JinQu: 洋红
    1: (255, 255, 0),    # C_ZhenLiaoShi: 青色
    2: (0, 255, 0),      # B_TongDao: 绿色
    3: (0, 255, 255),    # A_DaTing: 黄色
    4: (0, 128, 255),    # P: 橙色
    5: (0, 0, 0),        # background: 黑色
    6: (255, 0, 0),      # Bucket: 蓝色
}

BACKGROUND_ID = 5
BUCKET_ID = 6
VALID_IDS = set(CLASS_NAMES.keys())
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Semantic Bucket Labeler"


@dataclass(frozen=True)
class DatasetPaths:
    images_dir: Path
    source_masks_dir: Path
    output_masks_dir: Path
    output_overlays_dir: Path


@dataclass(frozen=True)
class Sample:
    image_path: Path
    source_mask_path: Path
    output_mask_path: Path
    output_overlay_path: Path
    relative_image_path: Path


def resolve_from_script(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def derive_dataset_paths(images_dir: Path) -> DatasetPaths:
    """从 .../images/... 推导 .../masks/...、.../masks7/...、.../overlays7/...。"""
    parts = images_dir.parts
    image_indices = [i for i, part in enumerate(parts) if part == "images"]
    if not image_indices:
        raise ValueError(
            "输入路径中必须包含名为 'images' 的目录，例如：\n"
            "bev_sem_round2_merged/images/train/2026-06-28_01-05-54"
        )

    idx = image_indices[-1]
    dataset_root = Path(*parts[:idx])
    tail = Path(*parts[idx + 1 :])

    return DatasetPaths(
        images_dir=images_dir,
        source_masks_dir=dataset_root / "masks" / tail,
        output_masks_dir=dataset_root / "masks7" / tail,
        output_overlays_dir=dataset_root / "overlays7" / tail,
    )


def find_images(images_dir: Path) -> List[Path]:
    image_paths = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise RuntimeError(f"没有找到图片：{images_dir}")
    return image_paths


def build_samples(paths: DatasetPaths) -> List[Sample]:
    samples: List[Sample] = []
    for image_path in find_images(paths.images_dir):
        rel = image_path.relative_to(paths.images_dir)
        rel_png = rel.with_suffix(".png")
        samples.append(
            Sample(
                image_path=image_path,
                source_mask_path=paths.source_masks_dir / rel_png,
                output_mask_path=paths.output_masks_dir / rel_png,
                output_overlay_path=paths.output_overlays_dir / rel_png,
                relative_image_path=rel,
            )
        )
    return samples


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图片：{path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"无法读取 mask：{path}")

    if mask.ndim == 3:
        # 语义标签必须是单通道。只有三个通道完全相同时才允许降为单通道。
        if not (
            np.array_equal(mask[:, :, 0], mask[:, :, 1])
            and np.array_equal(mask[:, :, 0], mask[:, :, 2])
        ):
            raise RuntimeError(f"mask 不是单通道类别 ID 图：{path}, shape={mask.shape}")
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise RuntimeError(f"mask 维度错误：{path}, shape={mask.shape}")

    if mask.dtype != np.uint8:
        if np.min(mask) < 0 or np.max(mask) > 255:
            raise RuntimeError(f"mask 无法安全转换为 uint8：{path}, dtype={mask.dtype}")
        mask = mask.astype(np.uint8)

    ids = set(np.unique(mask).astype(int).tolist())
    illegal = ids - VALID_IDS
    if illegal:
        raise RuntimeError(
            f"mask 含非法类别 ID：{path}\n"
            f"实际 ID={sorted(ids)}，允许 ID={sorted(VALID_IDS)}"
        )
    return mask


def write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    if not ok:
        raise RuntimeError(f"保存失败：{path}")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, bgr in CLASS_COLORS_BGR.items():
        color[mask == class_id] = bgr
    return color


def make_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"图片和 mask 尺寸不一致：image={image.shape[:2]}, mask={mask.shape[:2]}"
        )

    overlay = image.copy()
    color = colorize_mask(mask)

    # background 不覆盖原图；其余类别按颜色半透明叠加。
    foreground = mask != BACKGROUND_ID
    if np.any(foreground):
        blended = cv2.addWeighted(image, 1.0 - alpha, color, alpha, 0.0)
        overlay[foreground] = blended[foreground]

    # 为 Bucket 画一圈清晰边界，便于检查；不改变 mask。
    bucket_binary = (mask == BUCKET_ID).astype(np.uint8) * 255
    if np.any(bucket_binary):
        contours, _ = cv2.findContours(
            bucket_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, CLASS_COLORS_BGR[BUCKET_ID], 1)

    return overlay


def validate_pair(image: np.ndarray, mask: np.ndarray, sample: Sample) -> None:
    if image.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            "图片与 mask 尺寸不一致：\n"
            f"image: {sample.image_path} shape={image.shape[:2]}\n"
            f"mask : {sample.source_mask_path} shape={mask.shape[:2]}"
        )


def initialize_outputs(samples: List[Sample], reset_all: bool) -> None:
    """保证每一张图片都有 masks7 和 overlays7。"""
    print("=" * 72)
    print("初始化 masks7 / overlays7")
    print(f"样本数量: {len(samples)}")
    print(f"覆盖已有 masks7: {reset_all}")
    print("=" * 72)

    missing_masks: List[Path] = []
    for sample in samples:
        if not sample.source_mask_path.exists() and not sample.output_mask_path.exists():
            missing_masks.append(sample.source_mask_path)

    if missing_masks:
        preview = "\n".join(str(p) for p in missing_masks[:20])
        more = "" if len(missing_masks) <= 20 else f"\n... 另外还有 {len(missing_masks) - 20} 个"
        raise RuntimeError(
            f"有 {len(missing_masks)} 张图片找不到对应原始 mask：\n{preview}{more}"
        )

    for i, sample in enumerate(samples, start=1):
        image = read_image(sample.image_path)

        if reset_all or not sample.output_mask_path.exists():
            mask = read_mask(sample.source_mask_path)
            validate_pair(image, mask, sample)
            write_png(sample.output_mask_path, mask)
        else:
            mask = read_mask(sample.output_mask_path)
            if image.shape[:2] != mask.shape[:2]:
                raise RuntimeError(
                    f"已有 masks7 尺寸错误：{sample.output_mask_path}, "
                    f"image={image.shape[:2]}, mask={mask.shape[:2]}"
                )

        overlay = make_overlay(image, mask)
        write_png(sample.output_overlay_path, overlay)

        if i == 1 or i % 100 == 0 or i == len(samples):
            print(f"[{i:>5}/{len(samples)}] {sample.relative_image_path}")

    print("初始化完成：所有图片均已有 masks7 和 overlays7。")


class SemanticEditor:
    def __init__(
        self,
        samples: List[Sample],
        scale: int,
        brush_radius: int,
        start_index: int = 0,
    ) -> None:
        self.samples = samples
        self.scale = max(1, int(scale))
        self.brush_radius = max(1, int(brush_radius))
        self.index = min(max(0, start_index), len(samples) - 1)
        self.current_class = BUCKET_ID

        self.image: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.dirty = False

        self.drawing = False
        self.draw_class = BUCKET_ID
        self.last_point: Optional[Tuple[int, int]] = None
        self.undo_stack: List[np.ndarray] = []
        self.max_undo = 20
        self.status_message = ""

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        self.load_current()

    @property
    def sample(self) -> Sample:
        return self.samples[self.index]

    def push_undo(self) -> None:
        assert self.mask is not None
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def load_current(self) -> None:
        self.image = read_image(self.sample.image_path)
        self.mask = read_mask(self.sample.output_mask_path)
        if self.image.shape[:2] != self.mask.shape[:2]:
            raise RuntimeError(
                f"图片与 masks7 尺寸不一致：{self.sample.relative_image_path}"
            )
        self.dirty = False
        self.drawing = False
        self.last_point = None
        self.undo_stack.clear()
        self.status_message = "loaded"

    def save_current(self, reason: str = "saved") -> None:
        assert self.image is not None and self.mask is not None
        ids = set(np.unique(self.mask).astype(int).tolist())
        illegal = ids - VALID_IDS
        if illegal:
            raise RuntimeError(f"当前 mask 含非法 ID：{sorted(illegal)}")

        write_png(self.sample.output_mask_path, self.mask.astype(np.uint8))
        overlay = make_overlay(self.image, self.mask)
        write_png(self.sample.output_overlay_path, overlay)
        self.dirty = False
        self.status_message = reason
        print(
            f"[SAVE {self.index + 1}/{len(self.samples)}] "
            f"{self.sample.relative_image_path} | ids={sorted(ids)}"
        )

    def reset_to_source(self) -> None:
        assert self.image is not None
        source = read_mask(self.sample.source_mask_path)
        if source.shape != self.image.shape[:2]:
            raise RuntimeError("原始 mask 尺寸与图片不一致")
        self.push_undo()
        self.mask = source.copy()
        self.dirty = True
        self.status_message = "reset from original masks"

    def draw_at(self, point: Tuple[int, int]) -> None:
        assert self.mask is not None
        x, y = point
        h, w = self.mask.shape
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        if self.last_point is None:
            cv2.circle(
                self.mask,
                (x, y),
                self.brush_radius,
                int(self.draw_class),
                thickness=-1,
                lineType=cv2.LINE_8,
            )
        else:
            cv2.line(
                self.mask,
                self.last_point,
                (x, y),
                int(self.draw_class),
                thickness=self.brush_radius * 2,
                lineType=cv2.LINE_8,
            )
            cv2.circle(
                self.mask,
                (x, y),
                self.brush_radius,
                int(self.draw_class),
                thickness=-1,
                lineType=cv2.LINE_8,
            )

        self.last_point = (x, y)
        self.dirty = True
        self.status_message = (
            f"paint {self.draw_class}:{CLASS_NAMES[self.draw_class]}"
        )

    def screen_to_image(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        assert self.image is not None
        h, w = self.image.shape[:2]
        panel_w = w * self.scale
        panel_h = h * self.scale
        if x < 0 or y < 0 or x >= panel_w or y >= panel_h:
            return None
        return x // self.scale, y // self.scale

    def on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        point = self.screen_to_image(x, y)

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags)
            if delta > 0:
                self.brush_radius = min(100, self.brush_radius + 1)
            elif delta < 0:
                self.brush_radius = max(1, self.brush_radius - 1)
            self.status_message = f"brush={self.brush_radius}"
            return

        if event == cv2.EVENT_LBUTTONDOWN and point is not None:
            self.push_undo()
            self.drawing = True
            self.draw_class = self.current_class
            self.last_point = None
            self.draw_at(point)

        elif event == cv2.EVENT_RBUTTONDOWN and point is not None:
            self.push_undo()
            self.drawing = True
            self.draw_class = BACKGROUND_ID
            self.last_point = None
            self.draw_at(point)

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and point is not None:
            self.draw_at(point)

        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            if self.drawing and point is not None:
                self.draw_at(point)
            self.drawing = False
            self.last_point = None

    def undo(self) -> None:
        if not self.undo_stack:
            self.status_message = "undo stack empty"
            return
        self.mask = self.undo_stack.pop()
        self.dirty = True
        self.status_message = "undo"

    def change_index(self, new_index: int) -> None:
        # 无论是否修改，切换前都保存，保证输出始终存在并同步。
        self.save_current(reason="auto-saved")
        self.index = new_index % len(self.samples)
        self.load_current()

    def render(self) -> np.ndarray:
        assert self.image is not None and self.mask is not None
        overlay = make_overlay(self.image, self.mask)
        color_mask = colorize_mask(self.mask)

        h, w = self.image.shape[:2]
        overlay_big = cv2.resize(
            overlay, (w * self.scale, h * self.scale), interpolation=cv2.INTER_LINEAR
        )
        mask_big = cv2.resize(
            color_mask, (w * self.scale, h * self.scale), interpolation=cv2.INTER_NEAREST
        )

        top = np.hstack([overlay_big, mask_big])
        footer_h = 150
        canvas = np.zeros((top.shape[0] + footer_h, top.shape[1], 3), dtype=np.uint8)
        canvas[: top.shape[0]] = top

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "IMAGE + OVERLAY (editable)", (10, 24), font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "COLOR MASK", (w * self.scale + 10, 24), font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

        y0 = top.shape[0] + 22
        current_name = CLASS_NAMES[self.current_class]
        info = (
            f"[{self.index + 1}/{len(self.samples)}] {self.sample.relative_image_path} | "
            f"class={self.current_class}:{current_name} | brush={self.brush_radius} | "
            f"dirty={self.dirty} | {self.status_message}"
        )
        cv2.putText(canvas, info, (10, y0), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        help1 = "LMB paint | RMB background(5) | 0-6 class | S save | N/D next | P/A prev | U undo"
        help2 = "[ ] brush | mouse wheel brush | R reload masks7 | O reset from original masks | C clear | Q/ESC save+quit"
        cv2.putText(canvas, help1, (10, y0 + 24), font, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(canvas, help2, (10, y0 + 46), font, 0.45, (210, 210, 210), 1, cv2.LINE_AA)

        # 类别图例，确保数字、文字、颜色一一对应。
        legend_y = y0 + 72
        x = 10
        for class_id in range(7):
            color = CLASS_COLORS_BGR[class_id]
            cv2.rectangle(canvas, (x, legend_y - 12), (x + 18, legend_y + 4), color, -1)
            text = f"{class_id}:{CLASS_NAMES[class_id]}"
            cv2.putText(canvas, text, (x + 24, legend_y + 2), font, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            x += 150
            if x + 150 > canvas.shape[1]:
                x = 10
                legend_y += 25

        return canvas

    def run(self) -> None:
        print("\n操作：")
        print("  左键          绘制当前类别")
        print("  右键          擦除为 background=5")
        print("  0~6           选择类别，默认 6=Bucket")
        print("  S             手动保存")
        print("  N/D           下一张（自动保存当前图）")
        print("  P/A           上一张（自动保存当前图）")
        print("  U             撤销一次笔画")
        print("  [ / ]         减小/增大画笔")
        print("  鼠标滚轮      调整画笔")
        print("  R             从当前 masks7 重新加载")
        print("  O             恢复为原始 masks 标签")
        print("  C             整张清空为 background=5")
        print("  Q / Esc       自动保存并退出\n")

        while True:
            canvas = self.render()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if ord("0") <= key <= ord("6"):
                self.current_class = key - ord("0")
                self.status_message = (
                    f"selected {self.current_class}:{CLASS_NAMES[self.current_class]}"
                )

            elif key in (ord("s"), ord("S")):
                self.save_current(reason="manual-saved")

            elif key in (ord("n"), ord("N"), ord("d"), ord("D")):
                self.change_index(self.index + 1)

            elif key in (ord("p"), ord("P"), ord("a"), ord("A")):
                self.change_index(self.index - 1)

            elif key in (ord("u"), ord("U")):
                self.undo()

            elif key == ord("["):
                self.brush_radius = max(1, self.brush_radius - 1)
                self.status_message = f"brush={self.brush_radius}"

            elif key == ord("]"):
                self.brush_radius = min(100, self.brush_radius + 1)
                self.status_message = f"brush={self.brush_radius}"

            elif key in (ord("r"), ord("R")):
                self.load_current()
                self.status_message = "reloaded masks7"

            elif key in (ord("o"), ord("O")):
                self.reset_to_source()

            elif key in (ord("c"), ord("C")):
                assert self.mask is not None
                self.push_undo()
                self.mask.fill(BACKGROUND_ID)
                self.dirty = True
                self.status_message = "cleared to background=5"

            elif key in (ord("q"), ord("Q"), 27):
                self.save_current(reason="exit-auto-saved")
                break

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在现有语义 mask 上追加 class 6=Bucket，并输出 masks7/overlays7。"
    )
    parser.add_argument(
        "--images",
        required=True,
        help=(
            "待标注图片目录。相对路径按脚本所在目录解析，例如："
            "bev_sem_round2_merged/images/train/2026-06-28_01-05-54"
        ),
    )
    parser.add_argument("--scale", type=int, default=2, help="界面放大倍数，默认 2")
    parser.add_argument("--brush", type=int, default=5, help="初始画笔半径，默认 5 像素")
    parser.add_argument(
        "--start",
        default="",
        help="从文件相对路径中包含该字符串的第一张图片开始",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="用原始 masks 覆盖已有 masks7。谨慎使用，会清除已有 Bucket 标注。",
    )
    return parser.parse_args()


def find_start_index(samples: List[Sample], keyword: str) -> int:
    if not keyword:
        return 0
    for i, sample in enumerate(samples):
        if keyword in sample.relative_image_path.as_posix():
            return i
    raise RuntimeError(f"--start 未匹配任何文件：{keyword}")


def main() -> int:
    args = parse_args()
    images_dir = resolve_from_script(args.images)
    if not images_dir.is_dir():
        raise RuntimeError(f"图片目录不存在：{images_dir}")

    paths = derive_dataset_paths(images_dir)
    samples = build_samples(paths)

    print("=" * 72)
    print("Semantic Bucket Labeler")
    print(f"脚本目录       : {SCRIPT_DIR}")
    print(f"图片目录       : {paths.images_dir}")
    print(f"原始 masks     : {paths.source_masks_dir}")
    print(f"输出 masks7    : {paths.output_masks_dir}")
    print(f"输出 overlays7 : {paths.output_overlays_dir}")
    print("类别映射：")
    for class_id in range(7):
        print(f"  {class_id} = {CLASS_NAMES[class_id]}")
    print("=" * 72)

    initialize_outputs(samples, reset_all=args.reset_all)
    start_index = find_start_index(samples, args.start)

    editor = SemanticEditor(
        samples=samples,
        scale=args.scale,
        brush_radius=args.brush,
        start_index=start_index,
    )
    editor.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
