"""
视频还原：在原视频上绘制检测结果
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.utils import (
    detect_color_ellipses,
    filter_trajectory,
    load_config,
    get_data_dir,
)


def main():
    parser = argparse.ArgumentParser(
        description="Render detection results onto the original video"
    )
    parser.add_argument("--video", required=True, help="Input video file path")
    parser.add_argument(
        "--output",
        help="Output video path (default: output/videos/detection_result.mp4)",
    )
    parser.add_argument(
        "--show-trail",
        action="store_true",
        help="Show trajectory trails (recent 30 frames)",
    )
    parser.add_argument(
        "--filter-motion",
        action="store_true",
        help="Only show trajectories with motion amplitude > 5.0",
    )
    args = parser.parse_args()

    config = load_config()
    colors = config["colors"]
    color_map = {c["id"]: c for c in colors}
    roi = config.get("roi", None)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Unable to open video {args.video}")
        cap.release()
        return

    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {width}x{height}")
    print("Pass 1: Detecting all markers...")

    # 第一遍：收集所有检测结果
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

    print(f"Detection complete: {frame_idx} frames")

    # 过滤轨迹
    print("\nFiltering trajectories...")
    filtered_trajectories = {}
    motion_amplitudes = {}

    for color_id, trajectory in all_detections.items():
        if not trajectory:
            continue

        filtered = filter_trajectory(trajectory)
        if filtered:
            # 计算运动幅度
            positions = np.array([(t[1], t[2]) for t in filtered])
            motion_amplitude = np.std(positions, axis=0).mean()

            color_name = color_map[color_id]["name"]
            print(
                f"  {color_name} (ID {color_id}): {len(trajectory)} -> {len(filtered)} points, motion amplitude: {motion_amplitude:.1f}"
            )

            # 如果需要过滤运动幅度
            if args.filter_motion and motion_amplitude <= 5.0:
                print("    -> Filtered (motion amplitude < 5)")
                continue

            filtered_trajectories[color_id] = filtered
            motion_amplitudes[color_id] = motion_amplitude

    if not filtered_trajectories:
        print("No valid trajectories detected, outputting original video")

    # 转换为字典格式：{color_id: {frame_idx: (cx, cy)}}
    trajectory_dict = {}
    for color_id, traj in filtered_trajectories.items():
        trajectory_dict[color_id] = {t[0]: (t[1], t[2]) for t in traj}

    # 准备输出视频
    if args.output:
        output_path = args.output
    else:
        # 提取输入视频文件名（不含扩展名）
        video_basename = os.path.splitext(os.path.basename(args.video))[0]
        output_dir = str(get_data_dir("processed/videos"))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"detection_{video_basename}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 第二遍：绘制检测结果
    print("\nGenerating detection video...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0

    # 用于保存轨迹历史（显示尾迹）
    if args.show_trail:
        trail_history = {cid: [] for cid in filtered_trajectories.keys()}
        trail_length = 30  # 显示最近30帧的轨迹

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 创建绘制层
        overlay = frame.copy()

        # 绘制当前帧的检测点
        for color_id, traj_dict in trajectory_dict.items():
            if frame_idx in traj_dict:
                cx, cy = traj_dict[frame_idx]
                color_bgr = tuple(color_map[color_id]["bgr"])
                color_name = color_map[color_id]["name"]

                # 绘制圆点
                cv2.circle(overlay, (int(cx), int(cy)), 8, color_bgr, -1)
                cv2.circle(overlay, (int(cx), int(cy)), 10, (255, 255, 255), 2)

                # 绘制标签（英文）
                label = color_name  # 配置文件已改为英文
                label_pos = (int(cx) + 15, int(cy) - 10)

                # 绘制白色描边
                cv2.putText(
                    overlay,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )
                # 绘制彩色文字
                cv2.putText(
                    overlay,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_bgr,
                    1,
                )

                # 更新轨迹历史
                if args.show_trail:
                    trail_history[color_id].append((int(cx), int(cy)))
                    if len(trail_history[color_id]) > trail_length:
                        trail_history[color_id].pop(0)

        # 绘制轨迹尾迹
        if args.show_trail:
            for color_id, trail in trail_history.items():
                if len(trail) > 1:
                    color_bgr = tuple(color_map[color_id]["bgr"])
                    # 绘制连线
                    for i in range(1, len(trail)):
                        # 透明度随时间衰减
                        alpha = i / len(trail)
                        thickness = max(1, int(3 * alpha))
                        cv2.line(overlay, trail[i - 1], trail[i], color_bgr, thickness)

        # 混合绘制层
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # 添加信息栏
        info_text = f"Frame: {frame_idx}/{total_frames}"
        cv2.rectangle(frame, (5, 5), (250, 35), (0, 0, 0), -1)
        cv2.putText(
            frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # 写入视频
        out.write(frame)

        if frame_idx % 100 == 0:
            elapsed = time.time() - start_time
            progress = frame_idx / total_frames
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(
                f"  Progress: {frame_idx}/{total_frames} ({progress * 100:.1f}%), ETA: {eta:.1f}s"
            )

    cap.release()
    out.release()

    print(f"\n✓ Detection video saved to: {output_path}")
    print(f"  Contains {len(filtered_trajectories)} trajectories")
    print(f"  Total processing time: {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
