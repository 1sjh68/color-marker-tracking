"""
导出轨迹数据到CSV文件
"""

import argparse
import csv
import os

import cv2
from utils import detect_color_ellipses, filter_trajectory, load_config


def main():
    parser = argparse.ArgumentParser(description="Export trajectory data to CSV")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument(
        "--output", help="Output CSV file path (default: output/trajectories/trajectory_filename.csv)"
    )
    parser.add_argument(
        "--color-id", type=int, help="Export specific color ID only (optional)"
    )
    args = parser.parse_args()

    config = load_config()
    colors = config["colors"]
    roi = config.get("roi", None)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Unable to open video {args.video}")
        cap.release()  # Bug #7 修复：释放资源
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {total_frames} frames")
    print("Processing...")

    # 收集所有检测结果
    all_detections = {c["id"]: [] for c in colors}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detected = detect_color_ellipses(frame, colors, roi=roi)
        for color_id, cx, cy in detected:
            all_detections[color_id].append((frame_idx, cx, cy))

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    print(f"Detection complete: {frame_idx} frames")

    # 过滤轨迹
    print("\nFiltering trajectories...")
    filtered_trajectories = {}

    for color_id, trajectory in all_detections.items():
        if not trajectory:
            continue

        filtered = filter_trajectory(trajectory)
        if filtered:
            filtered_trajectories[color_id] = filtered
            color_name = next(
                (c["name"] for c in colors if c["id"] == color_id), f"Color{color_id}"
            )
            print(
                f"  {color_name} (ID {color_id}): {len(trajectory)} -> {len(filtered)} points"
            )

    # 选择要导出的轨迹
    if args.color_id is not None:
        # 导出指定颜色
        if args.color_id in filtered_trajectories:
            export_trajectories = {args.color_id: filtered_trajectories[args.color_id]}
            color_name = next(
                (c["name"] for c in colors if c["id"] == args.color_id),
                f"Color{args.color_id}",
            )
            print(f"\nExporting specified trajectory: {color_name} (ID {args.color_id})")
        else:
            print(f"\nError: Color ID {args.color_id} has no valid trajectory")
            return
    else:
        # 导出所有轨迹
        export_trajectories = filtered_trajectories
        total_points = sum(len(traj) for traj in export_trajectories.values())
        print(
            f"\nExporting all trajectories: {len(export_trajectories)} colors, {total_points} points total"
        )
        for cid in export_trajectories:
            color_name = next(
                (c["name"] for c in colors if c["id"] == cid), f"Color{cid}"
            )
            print(f"  {color_name} (ID {cid}): {len(export_trajectories[cid])} points")

    # 准备输出文件
    if args.output:
        output_path = args.output
    else:
        # 提取输入视频文件名（不含扩展名）
        video_basename = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "output", "trajectories"
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"trajectory_{video_basename}.csv")

    # 写入CSV（合并所有轨迹，按帧号排序）
    all_points = []
    for color_id, trajectory in export_trajectories.items():
        for frame, cx, cy in trajectory:
            all_points.append((frame, color_id, cx, cy))

    # 按帧号排序
    all_points.sort(key=lambda x: x[0])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "color_id", "u", "v"])
        for frame, color_id, cx, cy in all_points:
            writer.writerow([frame, color_id, cx, cy])

    print(f"\nDone! Trajectory data exported to: {output_path}")
    print(f"Total points: {len(all_points)}")
    print("\nData format: frame_idx, color_id, u, v")


if __name__ == "__main__":
    main()
