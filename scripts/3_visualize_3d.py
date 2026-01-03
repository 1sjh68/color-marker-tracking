"""
3D轨迹可视化
"""

import argparse
import os
import platform

import cv2
import matplotlib.pyplot as plt
import numpy as np
from utils import detect_color_ellipses, filter_trajectory, load_config


# 配置中文字体以避免警告
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    system = platform.system()
    if system == "Windows":
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    elif system == "Darwin":  # macOS
        plt.rcParams["font.sans-serif"] = [
            "PingFang SC",
            "Heiti SC",
            "Arial Unicode MS",
        ]
    elif system == "Linux":
        plt.rcParams["font.sans-serif"] = [
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "DejaVu Sans",
        ]

    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 设置中文字体
setup_chinese_font()


def main():
    parser = argparse.ArgumentParser(description="3D Trajectory Visualization")
    parser.add_argument("--video", required=True, help="Video file path")
    args = parser.parse_args()

    config = load_config()
    colors = config["colors"]
    color_names = {c["id"]: c["name"] for c in colors}
    roi = config.get("roi", None)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Unable to open video {args.video}")
        cap.release()  # Bug #7 修复：释放资源
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {total_frames} frames")
    print("Processing...")

    # 收集所有检测结果（不使用预测，完全基于空间连续性过滤）
    all_detections = {c["id"]: [] for c in colors}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 独立检测每一帧，不使用上一帧位置预测
        detected = detect_color_ellipses(frame, colors, roi=roi)

        for color_id, cx, cy in detected:
            all_detections[color_id].append((frame_idx, cx, cy))

        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    print(f"Detection complete: {frame_idx} frames")

    # 第二遍：过滤轨迹 + 运动幅度过滤
    print("\nFiltering trajectories...")
    temp_trajectories = {}
    motion_amplitudes = {}

    for color_id, trajectory in all_detections.items():
        filtered = filter_trajectory(trajectory)  # 使用默认参数
        if filtered:
            # 计算运动幅度
            positions = np.array([(t[1], t[2]) for t in filtered])
            motion_amplitude = np.std(positions, axis=0).mean()

            temp_trajectories[color_id] = filtered
            motion_amplitudes[color_id] = motion_amplitude
            print(
                f"  Color ID {color_id}: {len(trajectory)} -> {len(filtered)} points, motion: {motion_amplitude:.1f}"
            )

    # 按运动幅度过滤（使用固定阈值）
    if motion_amplitudes:
        motion_threshold = 5.0  # 降至5以保留所有颜色（包括运动较小的参考点）
        filtered_trajectories = {
            cid: traj
            for cid, traj in temp_trajectories.items()
            if motion_amplitudes[cid] > motion_threshold
        }
        print(f"\nKeeping trajectories with motion > {motion_threshold:.1f}")
    else:
        filtered_trajectories = temp_trajectories

    # 绘制3D图
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection="3d")

    colors_plot = ["red", "green", "blue", "yellow", "orange", "purple"]

    for color_id, traj in filtered_trajectories.items():
        if not traj:
            continue

        frames = [t[0] for t in traj]
        cx_vals = [t[1] for t in traj]
        cy_vals = [t[2] for t in traj]

        color = colors_plot[color_id % len(colors_plot)]
        label = f"ID {color_id} ({color_names.get(color_id, '?')}) - {len(traj)} points"

        ax.plot(
            frames,
            cx_vals,
            cy_vals,
            color=color,
            marker="o",
            markersize=3,
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
            label=label,
        )

    ax.set_xlabel("Frame Index (Time)", fontsize=12)
    ax.set_ylabel("X (Pixels u)", fontsize=12)
    ax.set_zlabel("Y (Pixels v)", fontsize=12)
    ax.set_title("Color Marker Trajectory 3D Plot (Filtered)", fontsize=14, pad=20)
    ax.legend(loc="upper left", fontsize=10)
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    # 保存图片
    # 提取输入视频文件名（不含扩展名）
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    output_dir = os.path.join(
        os.path.dirname(__file__), "..", "output", "visualizations"
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"trajectory_3d_{video_basename}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    print(f"\n3D Trajectory plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
