#!/usr/bin/env python3

import argparse
import os
import glob
from pathlib import Path

import cv2
import numpy as np
import torch


CLASS_NAMES = {
    0: "JinQu",
    1: "C_ZhenLiaoShi",
    2: "B_TongDao",
    3: "A_DaTing",
    4: "P",
    5: "background",
}

# 后面重定位要用的可走区域
TRAVERSABLE_IDS = [2, 3, 4]  # B_TongDao, A_DaTing, P


def collect_images(input_path):
    input_path = Path(input_path)

    if input_path.is_file():
        return [str(input_path)]

    if input_path.is_dir():
        exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths = []
        for ext in exts:
            paths.extend(glob.glob(str(input_path / ext)))
            paths.extend(glob.glob(str(input_path / ext.upper())))
        return sorted(paths)

    # 支持 glob，例如 "bev/*.png"
    return sorted(glob.glob(str(input_path)))


def unwrap_model_output(output):
    """
    兼容不同 TorchScript 导出形式。
    目标是拿到 [N, C, H, W] 的 logits。
    """
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (list, tuple)):
        # 常见情况：output[0] 是 logits
        for item in output:
            if isinstance(item, torch.Tensor) and item.ndim == 4:
                return item
        raise RuntimeError("Cannot find 4D tensor output in tuple/list model output.")

    if isinstance(output, dict):
        for key in ["out", "output", "logits", "pred"]:
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]

        for value in output.values():
            if isinstance(value, torch.Tensor) and value.ndim == 4:
                return value

        raise RuntimeError("Cannot find 4D tensor output in dict model output.")

    raise RuntimeError(f"Unsupported model output type: {type(output)}")


def preprocess_bgr(img_bgr, input_size=320):
    """
    输入 OpenCV BGR 图像，输出模型输入 tensor: [1, 3, 320, 320]

    这里默认：
      BGR -> RGB
      uint8 0~255 -> float32 0~1

    如果结果完全不对，再考虑模型是不是训练时用 BGR 或者有 mean/std。
    """
    if img_bgr.shape[0] != input_size or img_bgr.shape[1] != input_size:
        img_bgr = cv2.resize(img_bgr, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))       # HWC -> CHW
    x = np.expand_dims(x, axis=0)        # CHW -> NCHW
    x = torch.from_numpy(x)
    return x


def colorize_class_map(cls_map):
    """
    输出 BGR 可视化图。
    """
    colors = {
        0: (0, 0, 255),       # JinQu: red
        1: (255, 0, 0),       # C_ZhenLiaoShi: blue
        2: (0, 255, 0),       # B_TongDao: green
        3: (0, 255, 255),     # A_DaTing: yellow
        4: (255, 255, 255),   # P: white
        5: (0, 0, 0),         # background: black
    }

    vis = np.zeros((cls_map.shape[0], cls_map.shape[1], 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        vis[cls_map == cls_id] = color

    return vis


def make_overlay(img_bgr, sem_vis, alpha=0.45):
    img_bgr = cv2.resize(img_bgr, (sem_vis.shape[1], sem_vis.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(img_bgr, 1.0 - alpha, sem_vis, alpha, 0)


def save_legend(out_dir):
    legend_path = os.path.join(out_dir, "class_legend.txt")
    with open(legend_path, "w", encoding="utf-8") as f:
        for cls_id, name in CLASS_NAMES.items():
            f.write(f"{cls_id}: {name}\n")
        f.write("\nTraversable ids: 2, 3, 4 = B_TongDao, A_DaTing, P\n")


def infer_one(model, img_path, out_dir, device, input_size=320):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[WARN] failed to read image: {img_path}")
        return False

    x = preprocess_bgr(img_bgr, input_size=input_size).to(device)

    with torch.no_grad():
        output = model(x)
        logits = unwrap_model_output(output)

    if logits.ndim != 4:
        raise RuntimeError(f"Expected logits shape [N,C,H,W], got {tuple(logits.shape)}")

    # 如果输出不是 320x320，拉回 320x320
    if logits.shape[-2:] != (input_size, input_size):
        logits = torch.nn.functional.interpolate(
            logits,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )

    cls_map = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

    sem_vis = colorize_class_map(cls_map)

    traversable_mask = np.isin(cls_map, TRAVERSABLE_IDS).astype(np.uint8) * 255

    overlay = make_overlay(img_bgr, sem_vis)

    stem = Path(img_path).stem

    class_map_path = os.path.join(out_dir, "class_map", f"{stem}_class.png")
    sem_vis_path = os.path.join(out_dir, "semantic_vis", f"{stem}_sem.png")
    mask_path = os.path.join(out_dir, "traversable_mask", f"{stem}_trav.png")
    overlay_path = os.path.join(out_dir, "overlay", f"{stem}_overlay.png")

    cv2.imwrite(class_map_path, cls_map)
    cv2.imwrite(sem_vis_path, sem_vis)
    cv2.imwrite(mask_path, traversable_mask)
    cv2.imwrite(overlay_path, overlay)

    unique_ids, counts = np.unique(cls_map, return_counts=True)
    stat = ", ".join(
        f"{int(i)}:{CLASS_NAMES.get(int(i), 'unknown')}={int(c)}"
        for i, c in zip(unique_ids, counts)
    )

    print(f"[OK] {img_path}")
    print(f"     classes: {stat}")

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="Path to exp.torchscript")
    parser.add_argument("--input", required=True, help="BEV image file, folder, or glob")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--input-size", type=int, default=320, help="Model input size, default 320")

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Fallback to CPU.")
        device = "cpu"

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "class_map"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "semantic_vis"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "traversable_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.output, "overlay"), exist_ok=True)

    save_legend(args.output)

    print("========================================")
    print("TorchScript BEV semantic inference")
    print(f"Model:      {args.model}")
    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Device:     {device}")
    print(f"Input size: {args.input_size}")
    print("========================================")

    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    image_paths = collect_images(args.input)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found: {args.input}")

    ok_count = 0
    for img_path in image_paths:
        ok = infer_one(
            model=model,
            img_path=img_path,
            out_dir=args.output,
            device=device,
            input_size=args.input_size,
        )
        ok_count += int(ok)

    print("========================================")
    print(f"[DONE] {ok_count}/{len(image_paths)} images processed.")
    print(f"Results saved to: {args.output}")
    print("========================================")


if __name__ == "__main__":
    main()