#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag folder path")
    parser.add_argument("--topic", default="/ros_image/jpeg")
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max_images", type=int, default=300, help="0 means no limit")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    storage_options = rosbag2_py.StorageOptions(
        uri=args.bag,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    print("Topics in bag:")
    for name, msg_type in type_map.items():
        print(f"  {name}: {msg_type}")

    if args.topic not in type_map:
        raise RuntimeError(f"Topic not found: {args.topic}")

    msg_type = get_message(type_map[args.topic])

    frame_idx = 0
    saved_idx = 0

    while reader.has_next():
        topic, data, timestamp = reader.read_next()

        if topic != args.topic:
            continue

        if frame_idx % args.stride == 0:
            msg = deserialize_message(data, msg_type)

            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None:
                print(f"[WARN] decode failed at frame {frame_idx}")
            else:
                save_name = f"frame_{saved_idx:06d}_{timestamp}.jpg"
                save_path = os.path.join(args.out, save_name)
                cv2.imwrite(save_path, img)

                if saved_idx % 20 == 0:
                    print(f"saved {saved_idx}: {save_path}")

                saved_idx += 1

                if args.max_images > 0 and saved_idx >= args.max_images:
                    break

        frame_idx += 1

    print("Done.")
    print(f"Total image messages read: {frame_idx}")
    print(f"Images saved: {saved_idx}")
    print(f"Output folder: {args.out}")


if __name__ == "__main__":
    main()