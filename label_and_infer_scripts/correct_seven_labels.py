#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
七类语义分割标签修正工具。

固定目录结构（均相对于脚本所在目录）：

图片：
    bev_sem_round2_merged/images/<split>/<run>/...

原始七类标签：
    label_seven/classseventh/<split>/<run>/...

修正后标签：
    label_seven/corrected_seven/<split>/<run>/...

修正后 Overlay：
    label_seven/corrected_overlays/<split>/<run>/...

使用示例：
    python3 correct_seven_labels.py \
        --folder train/2026-06-28_01-05-54

核心保存协议：
1. 打开一张图片时，若 corrected_seven 中还没有对应标签，
   会立即把 classseventh 中的原标签复制到 corrected_seven，
   同时生成 corrected_overlays。
2. 按 N/D、P/A 切换前，无论是否修改，都会保存当前标签和 overlay。
3. 按 S 手动保存。
4. 按 Q/Esc 退出时自动保存。
5. 已有 corrected_seven 时，优先从 corrected_seven 继续修改。
6. 原始 classseventh 永远不修改。

类别协议：
    0 = JinQu
    1 = C_ZhenLiaoShi
    2 = B_TongDao
    3 = A_DaTing
    4 = P
    5 = background
    6 = Bucket
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

IMAGES_ROOT = SCRIPT_DIR / "bev_sem_round2_merged" / "images"
SOURCE_MASKS_ROOT = SCRIPT_DIR / "label_seven" / "classseventh"
OUTPUT_MASKS_ROOT = SCRIPT_DIR / "label_seven" / "corrected_seven"
OUTPUT_OVERLAYS_ROOT = SCRIPT_DIR / "label_seven" / "corrected_overlays"

CLASS_NAMES: Dict[int, str] = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
    6: "Bucket",
}

# OpenCV 使用 BGR。颜色只影响显示，不影响 mask 中的类别数字。
CLASS_COLORS_BGR: Dict[int, Tuple[int, int, int]] = {
    0: (255, 0, 255),    # JinQu：洋红
    1: (255, 255, 0),    # C_ZhenLiaoShi：青色
    2: (0, 255, 0),      # B_TongDao：绿色
    3: (0, 255, 255),    # A_DaTing：黄色
    4: (0, 128, 255),    # P：橙色
    5: (0, 0, 0),        # background：黑色
    6: (0, 0, 255),      # Bucket：红色
}

BACKGROUND_ID = 5
BUCKET_ID = 6
VALID_IDS = set(CLASS_NAMES)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
WINDOW_NAME = "Seven-Class Semantic Label Corrector"


@dataclass(frozen=True)
class Sample:
    image_path: Path
    source_mask_path: Path
    corrected_mask_path: Path
    corrected_overlay_path: Path
    relative_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="修正完整七类语义分割标签，并保存到 corrected_seven。"
    )
    parser.add_argument(
        "--folder",
        required=True,
        help=(
            "需要处理的相对目录，例如："
            "train/2026-06-28_01-05-54 或 "
            "val/2026-06-28_01-05-54"
        ),
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        help="界面放大倍数，默认 2。",
    )
    parser.add_argument(
        "--brush",
        type=int,
        default=5,
        help="初始画笔半径，单位为原图像素，默认 5。",
    )
    parser.add_argument(
        "--start",
        default="",
        help="从相对文件路径中包含该字符串的第一张开始。",
    )
    parser.add_argument(
        "--reset-current-from-source",
        action="store_true",
        help=(
            "仅在启动时，将第一张当前图片恢复为 classseventh 原标签。"
            "不会批量覆盖其他 corrected_seven。"
        ),
    )
    return parser.parse_args()


def normalize_folder(folder_text: str) -> Path:
    folder = Path(folder_text.strip().strip("/"))

    if folder.is_absolute():
        raise ValueError(
            "--folder 必须使用相对路径，例如 "
            "train/2026-06-28_01-05-54"
        )

    if len(folder.parts) < 2:
        raise ValueError(
            "--folder 至少应包含 split/run，例如 "
            "train/2026-06-28_01-05-54"
        )

    if folder.parts[0] not in {"train", "val", "test"}:
        raise ValueError(
            "--folder 第一层必须是 train、val 或 test，"
            f"当前为：{folder.parts[0]}"
        )

    return folder


def find_images(root: Path) -> List[Path]:
    images = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not images:
        raise RuntimeError(f"目录内没有找到图片：{root}")

    return images


def build_samples(folder: Path) -> List[Sample]:
    image_dir = IMAGES_ROOT / folder
    source_mask_dir = SOURCE_MASKS_ROOT / folder
    corrected_mask_dir = OUTPUT_MASKS_ROOT / folder
    corrected_overlay_dir = OUTPUT_OVERLAYS_ROOT / folder

    if not image_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{image_dir}")

    if not source_mask_dir.is_dir():
        raise FileNotFoundError(
            f"原始七类标签目录不存在：{source_mask_dir}"
        )

    samples: List[Sample] = []
    missing_masks: List[Path] = []

    for image_path in find_images(image_dir):
        relative = image_path.relative_to(image_dir)
        relative_png = relative.with_suffix(".png")
        source_mask_path = source_mask_dir / relative_png

        if not source_mask_path.is_file():
            missing_masks.append(source_mask_path)
            continue

        samples.append(
            Sample(
                image_path=image_path,
                source_mask_path=source_mask_path,
                corrected_mask_path=corrected_mask_dir / relative_png,
                corrected_overlay_path=corrected_overlay_dir / relative_png,
                relative_path=relative,
            )
        )

    if missing_masks:
        preview = "\n".join(str(path) for path in missing_masks[:20])
        extra = ""
        if len(missing_masks) > 20:
            extra = (
                f"\n……另外还有 {len(missing_masks) - 20} 个缺失标签。"
            )
        raise RuntimeError(
            f"共有 {len(missing_masks)} 张图片缺少 classseventh 标签：\n"
            f"{preview}{extra}"
        )

    if not samples:
        raise RuntimeError(f"没有可处理样本：{folder}")

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
        if not (
            np.array_equal(mask[:, :, 0], mask[:, :, 1])
            and np.array_equal(mask[:, :, 0], mask[:, :, 2])
        ):
            raise RuntimeError(
                f"mask 不是单通道类别 ID 图：{path}，shape={mask.shape}"
            )
        mask = mask[:, :, 0]

    if mask.ndim != 2:
        raise RuntimeError(
            f"mask 维度错误：{path}，shape={mask.shape}"
        )

    if mask.dtype != np.uint8:
        min_value = int(mask.min())
        max_value = int(mask.max())
        if min_value < 0 or max_value > 255:
            raise RuntimeError(
                f"mask 无法转换为 uint8：{path}，dtype={mask.dtype}"
            )
        mask = mask.astype(np.uint8)

    ids = set(np.unique(mask).astype(int).tolist())
    illegal_ids = ids - VALID_IDS

    if illegal_ids:
        raise RuntimeError(
            f"mask 包含非法类别 ID：{path}\n"
            f"实际 ID={sorted(ids)}\n"
            f"允许 ID={sorted(VALID_IDS)}"
        )

    return mask


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"保存失败：{path}")


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)

    for class_id, bgr in CLASS_COLORS_BGR.items():
        color[mask == class_id] = bgr

    return color


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise RuntimeError(
            "图片与 mask 尺寸不一致："
            f"image={image.shape[:2]}, mask={mask.shape}"
        )

    overlay = image.copy()
    color_mask = colorize_mask(mask)

    # background=5 保留原图，其余类别进行半透明叠加。
    foreground = mask != BACKGROUND_ID
    if np.any(foreground):
        blended = cv2.addWeighted(
            image,
            1.0 - alpha,
            color_mask,
            alpha,
            0.0,
        )
        overlay[foreground] = blended[foreground]

    # Bucket=6 加粗边界，便于人工检查。
    bucket_binary = (mask == BUCKET_ID).astype(np.uint8) * 255
    if np.any(bucket_binary):
        contours, _ = cv2.findContours(
            bucket_binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            CLASS_COLORS_BGR[BUCKET_ID],
            1,
            lineType=cv2.LINE_AA,
        )

    return overlay


def find_start_index(samples: List[Sample], keyword: str) -> int:
    if not keyword:
        return 0

    for index, sample in enumerate(samples):
        if keyword in sample.relative_path.as_posix():
            return index

    raise RuntimeError(f"--start 未匹配任何文件：{keyword}")


class SevenClassEditor:
    def __init__(
        self,
        samples: List[Sample],
        scale: int,
        brush_radius: int,
        start_index: int,
        reset_current_from_source: bool,
    ) -> None:
        self.samples = samples
        self.scale = max(1, int(scale))
        self.brush_radius = max(1, int(brush_radius))
        self.index = min(max(start_index, 0), len(samples) - 1)

        # 默认选择 Bucket，便于主要修正新增类别。
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

        self.load_current(
            force_source=reset_current_from_source,
            save_on_load=True,
        )

    @property
    def sample(self) -> Sample:
        return self.samples[self.index]

    def push_undo(self) -> None:
        assert self.mask is not None

        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)

    def load_current(
        self,
        *,
        force_source: bool = False,
        save_on_load: bool = True,
    ) -> None:
        self.image = read_image(self.sample.image_path)

        if (
            not force_source
            and self.sample.corrected_mask_path.is_file()
        ):
            mask_path = self.sample.corrected_mask_path
            source_name = "corrected_seven"
        else:
            mask_path = self.sample.source_mask_path
            source_name = "classseventh"

        self.mask = read_mask(mask_path)

        if self.image.shape[:2] != self.mask.shape:
            raise RuntimeError(
                "图片与标签尺寸不一致：\n"
                f"图片：{self.sample.image_path} "
                f"shape={self.image.shape[:2]}\n"
                f"标签：{mask_path} shape={self.mask.shape}"
            )

        self.dirty = False
        self.drawing = False
        self.last_point = None
        self.undo_stack.clear()
        self.status_message = f"loaded from {source_name}"

        # 只要这张标签被读取并显示，就立即写入 corrected_seven。
        if save_on_load:
            self.save_current(reason=f"copied from {source_name}")

    def save_current(self, reason: str = "saved") -> None:
        assert self.image is not None
        assert self.mask is not None

        ids = set(np.unique(self.mask).astype(int).tolist())
        illegal_ids = ids - VALID_IDS
        if illegal_ids:
            raise RuntimeError(
                f"当前 mask 包含非法 ID：{sorted(illegal_ids)}"
            )

        write_png(
            self.sample.corrected_mask_path,
            self.mask.astype(np.uint8),
        )

        overlay = make_overlay(self.image, self.mask)
        write_png(
            self.sample.corrected_overlay_path,
            overlay,
        )

        self.dirty = False
        self.status_message = reason

        print(
            f"[SAVE {self.index + 1}/{len(self.samples)}] "
            f"{self.sample.relative_path.as_posix()} | "
            f"ids={sorted(ids)} | {reason}"
        )

    def reset_to_source(self) -> None:
        assert self.image is not None

        source_mask = read_mask(self.sample.source_mask_path)
        if source_mask.shape != self.image.shape[:2]:
            raise RuntimeError(
                "classseventh 原标签尺寸与图片不一致。"
            )

        self.push_undo()
        self.mask = source_mask.copy()
        self.dirty = True
        self.status_message = "reset from classseventh"

    def reload_corrected(self) -> None:
        if self.sample.corrected_mask_path.is_file():
            self.image = read_image(self.sample.image_path)
            self.mask = read_mask(self.sample.corrected_mask_path)
            self.dirty = False
            self.undo_stack.clear()
            self.status_message = "reloaded corrected_seven"
        else:
            self.load_current(force_source=False, save_on_load=True)

    def draw_at(self, point: Tuple[int, int]) -> None:
        assert self.mask is not None

        x, y = point
        height, width = self.mask.shape

        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))

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
            f"paint {self.draw_class}:"
            f"{CLASS_NAMES[self.draw_class]}"
        )

    def screen_to_image(
        self,
        x: int,
        y: int,
    ) -> Optional[Tuple[int, int]]:
        assert self.image is not None

        height, width = self.image.shape[:2]
        editable_width = width * self.scale
        editable_height = height * self.scale

        # 只有左侧 overlay 区域可编辑。
        if (
            x < 0
            or y < 0
            or x >= editable_width
            or y >= editable_height
        ):
            return None

        return x // self.scale, y // self.scale

    def on_mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        _param,
    ) -> None:
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

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and self.drawing
            and point is not None
        ):
            self.draw_at(point)

        elif event in (
            cv2.EVENT_LBUTTONUP,
            cv2.EVENT_RBUTTONUP,
        ):
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
        # 无论是否修改，离开当前图片前都保存。
        self.save_current(reason="auto-saved before navigation")

        self.index = new_index % len(self.samples)

        # 下一张一旦被读取，也立即复制到 corrected_seven。
        self.load_current(
            force_source=False,
            save_on_load=True,
        )

    def render(self) -> np.ndarray:
        assert self.image is not None
        assert self.mask is not None

        overlay = make_overlay(self.image, self.mask)
        color_mask = colorize_mask(self.mask)

        height, width = self.image.shape[:2]

        overlay_big = cv2.resize(
            overlay,
            (width * self.scale, height * self.scale),
            interpolation=cv2.INTER_LINEAR,
        )
        mask_big = cv2.resize(
            color_mask,
            (width * self.scale, height * self.scale),
            interpolation=cv2.INTER_NEAREST,
        )

        top = np.hstack([overlay_big, mask_big])

        footer_height = 155
        canvas = np.zeros(
            (top.shape[0] + footer_height, top.shape[1], 3),
            dtype=np.uint8,
        )
        canvas[: top.shape[0]] = top

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(
            canvas,
            "IMAGE + OVERLAY (editable)",
            (10, 24),
            font,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "SEVEN-CLASS COLOR MASK",
            (width * self.scale + 10, 24),
            font,
            0.58,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y0 = top.shape[0] + 22
        current_name = CLASS_NAMES[self.current_class]

        info = (
            f"[{self.index + 1}/{len(self.samples)}] "
            f"{self.sample.relative_path.as_posix()} | "
            f"class={self.current_class}:{current_name} | "
            f"brush={self.brush_radius} | "
            f"dirty={self.dirty} | "
            f"{self.status_message}"
        )

        cv2.putText(
            canvas,
            info,
            (10, y0),
            font,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        help_line_1 = (
            "LMB paint | RMB background(5) | 0-6 class | "
            "S save | N/D next | P/A prev | U undo"
        )
        help_line_2 = (
            "[ ] brush | wheel brush | R reload corrected | "
            "O reset classseventh | C clear background | Q/ESC save+quit"
        )

        cv2.putText(
            canvas,
            help_line_1,
            (10, y0 + 24),
            font,
            0.44,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            help_line_2,
            (10, y0 + 46),
            font,
            0.44,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        legend_y = y0 + 76
        legend_x = 10

        for class_id in range(7):
            color = CLASS_COLORS_BGR[class_id]

            cv2.rectangle(
                canvas,
                (legend_x, legend_y - 12),
                (legend_x + 18, legend_y + 4),
                color,
                thickness=-1,
            )

            cv2.putText(
                canvas,
                f"{class_id}:{CLASS_NAMES[class_id]}",
                (legend_x + 24, legend_y + 2),
                font,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            legend_x += 150
            if legend_x + 150 > canvas.shape[1]:
                legend_x = 10
                legend_y += 25

        return canvas

    def run(self) -> None:
        print("\n操作说明：")
        print("  左键          绘制当前类别")
        print("  右键          擦除为 background=5")
        print("  0～6          选择类别，默认 6=Bucket")
        print("  S             手动保存当前标签和 overlay")
        print("  N / D         下一张，自动保存当前图片")
        print("  P / A         上一张，自动保存当前图片")
        print("  U             撤销上一次笔画")
        print("  [ / ]         减小 / 增大画笔")
        print("  鼠标滚轮      调整画笔大小")
        print("  R             重新读取 corrected_seven")
        print("  O             恢复为原始 classseventh 标签")
        print("  C             整张清空为 background=5")
        print("  Q / Esc       自动保存并退出")
        print()
        print(
            "注意：图片一旦被显示，就会立即写入 corrected_seven 和 "
            "corrected_overlays；即使未修改并直接按 N，也会保留。"
        )
        print()

        while True:
            cv2.imshow(WINDOW_NAME, self.render())
            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if ord("0") <= key <= ord("6"):
                self.current_class = key - ord("0")
                self.status_message = (
                    f"selected {self.current_class}:"
                    f"{CLASS_NAMES[self.current_class]}"
                )

            elif key in (ord("s"), ord("S")):
                self.save_current(reason="manual-saved")

            elif key in (
                ord("n"),
                ord("N"),
                ord("d"),
                ord("D"),
            ):
                self.change_index(self.index + 1)

            elif key in (
                ord("p"),
                ord("P"),
                ord("a"),
                ord("A"),
            ):
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
                self.reload_corrected()

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


def main() -> int:
    args = parse_args()
    folder = normalize_folder(args.folder)

    samples = build_samples(folder)
    start_index = find_start_index(samples, args.start)

    print("=" * 78)
    print("Seven-Class Semantic Label Corrector")
    print(f"脚本目录       : {SCRIPT_DIR}")
    print(f"处理目录       : {folder}")
    print(f"图片目录       : {IMAGES_ROOT / folder}")
    print(f"原七类标签     : {SOURCE_MASKS_ROOT / folder}")
    print(f"修正后标签     : {OUTPUT_MASKS_ROOT / folder}")
    print(f"修正后 Overlay : {OUTPUT_OVERLAYS_ROOT / folder}")
    print(f"样本数量       : {len(samples)}")
    print("类别映射：")

    for class_id in range(7):
        print(
            f"  {class_id} = {CLASS_NAMES[class_id]} "
            f"| BGR={CLASS_COLORS_BGR[class_id]}"
        )

    print("=" * 78)

    editor = SevenClassEditor(
        samples=samples,
        scale=args.scale,
        brush_radius=args.brush,
        start_index=start_index,
        reset_current_from_source=args.reset_current_from_source,
    )
    editor.run()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
