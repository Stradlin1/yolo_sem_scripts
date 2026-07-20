#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def detect_storage_id(bag_dir: Path) -> str:
    if list(bag_dir.glob("*.db3")):
        return "sqlite3"
    if list(bag_dir.glob("*.mcap")):
        return "mcap"
    raise RuntimeError(f"No .db3 or .mcap file found in: {bag_dir}")


def extract_one_bag(bag_dir: Path, output_root: Path, topic_name: str, ext: str) -> int:
    bag_dir = bag_dir.resolve()
    output_dir = output_root / bag_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    storage_id = detect_storage_id(bag_dir)

    reader = rosbag2_py.SequentialReader()

    storage_options = rosbag2_py.StorageOptions(
        uri=str(bag_dir),
        storage_id=storage_id,
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if topic_name not in type_map:
        print(f"[SKIP] topic not found in {bag_dir.name}: {topic_name}")
        print("Available topics:")
        for t in topic_types:
            print(f"  {t.name} | {t.type}")
        return 0

    msg_type = get_message(type_map[topic_name])

    count = 0
    skipped = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic != topic_name:
            continue

        msg = deserialize_message(data, msg_type)

        img_array = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            skipped += 1
            continue

        filename = f"{count:06d}_{timestamp}.{ext}"
        save_path = output_dir / filename

        ok = cv2.imwrite(str(save_path), img)
        if ok:
            count += 1
        else:
            skipped += 1

        if count > 0 and count % 500 == 0:
            print(f"[{bag_dir.name}] saved {count} frames...")

    print("-" * 80)
    print(f"bag     : {bag_dir}")
    print(f"topic   : {topic_name}")
    print(f"output  : {output_dir}")
    print(f"saved   : {count}")
    print(f"skipped : {skipped}")
    print("-" * 80)

    return count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        default="/home/xhm/Desktop/relocate_ws/data/rosbags/raw",
        help="root folder containing rosbag folders",
    )

    parser.add_argument(
        "--output-root",
        default="/home/xhm/Desktop/relocate_ws/data/extracted/raw_all_frames",
        help="output root folder",
    )

    parser.add_argument(
        "--topic",
        default="/ros_image/jpeg",
        help="image topic name",
    )

    parser.add_argument(
        "--ext",
        default="jpg",
        choices=["jpg", "png"],
        help="output image extension",
    )

    parser.add_argument(
        "--bag-name",
        default="",
        help="only extract one bag folder by name, empty means all bags",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise RuntimeError(f"input root does not exist: {input_root}")

    if args.bag_name:
        bag_dirs = [input_root / args.bag_name]
    else:
        bag_dirs = sorted(
            p for p in input_root.iterdir()
            if p.is_dir() and (list(p.glob("*.db3")) or list(p.glob("*.mcap")))
        )

    if not bag_dirs:
        raise RuntimeError(f"No rosbag folders found in: {input_root}")

    print("=" * 80)
    print("Extract ROS2 compressed images")
    print(f"input root : {input_root}")
    print(f"output root: {output_root}")
    print(f"topic      : {args.topic}")
    print(f"bags       : {len(bag_dirs)}")
    print("=" * 80)

    total = 0

    for bag_dir in bag_dirs:
        if not bag_dir.exists():
            print(f"[SKIP] bag folder does not exist: {bag_dir}")
            continue

        total += extract_one_bag(
            bag_dir=bag_dir,
            output_root=output_root,
            topic_name=args.topic,
            ext=args.ext,
        )

    print("=" * 80)
    print(f"ALL DONE. total saved frames: {total}")
    print(f"output root: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
