"""
彩色圆点检测与跟踪（带智能降噪）
"""

import argparse
import time

import cv2
import numpy as np
from utils import detect_color_ellipses, filter_trajectory, load_config


def main():
    parser = argparse.ArgumentParser(description="Color dot detection and tracking")
    parser.add_argument("--video", required=True, help="Video file path")
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
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info: {total_frames} frames, {fps:.2f} fps")
    print("Processing...")

    cv2.namedWindow("Color Dots Tracking", cv2.WINDOW_NORMAL)
    _w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _mw, _mh = 1280, 720
    _scale = min(_mw / max(1, _w), _mh / max(1, _h))
    _dw, _dh = (int(_w * _scale), int(_h * _scale)) if _scale > 0 else (_w, _h)
    if _dw > 0 and _dh > 0:
        cv2.resizeWindow("Color Dots Tracking", _dw, _dh)

    # 第一遍：收集所有检测结果（不使用预测，完全基于空间连续性过滤）
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

    print(f"Detection complete: {frame_idx} frames")

    # 第二遍：过滤轨迹 + 计算运动幅度
    print("\nFiltering trajectories...")
    temp_trajectories = {}
    motion_amplitudes = {}

    for color_id, trajectory in all_detections.items():
        if not trajectory:
            continue

        filtered = filter_trajectory(trajectory)
        if filtered:
            # 计算运动幅度（位置标准差）
            positions = np.array([(t[1], t[2]) for t in filtered])
            motion_amplitude = np.std(positions, axis=0).mean()

            temp_trajectories[color_id] = {t[0]: (t[1], t[2]) for t in filtered}
            motion_amplitudes[color_id] = motion_amplitude

            color_name = next(
                (c["name"] for c in colors if c["id"] == color_id), f"Color{color_id}"
            )
            print(
                f"  {color_name} (ID {color_id}): {len(trajectory)} -> {len(filtered)} points, motion amplitude: {motion_amplitude:.1f}"
            )

    # 按运动幅度 + 连续性过滤：保留运动幅度较大且连续性好的轨迹
    if motion_amplitudes:
        motion_threshold = 5.0  # 运动幅度阈值（标准差）- 降至5以保留所有颜色
        min_points = 50  # 最少连续点数（降低以保留更多短轨迹）
        print(
            f"\nFilter thresholds: motion amplitude > {motion_threshold:.1f}, min points >= {min_points}"
        )

        # 同时满足运动幅度和连续性要求
        filtered_trajectories = {
            cid: traj
            for cid, traj in temp_trajectories.items()
            if motion_amplitudes[cid] > motion_threshold and len(traj) >= min_points
        }

        print("\nFilter results:")
        for cid in filtered_trajectories:
            cname = next((c["name"] for c in colors if c["id"] == cid), f"Color{cid}")
            coverage = len(filtered_trajectories[cid]) / total_frames * 100
            print(
                f"  ✅ Kept: {cname} (ID {cid}), points: {len(filtered_trajectories[cid])}, motion amplitude: {motion_amplitudes[cid]:.1f}, coverage: {coverage:.1f}%"
            )

        for cid in temp_trajectories:
            if cid not in filtered_trajectories:
                cname = next(
                    (c["name"] for c in colors if c["id"] == cid), f"Color{cid}"
                )
                points = len(temp_trajectories[cid])
                reason = ""
                if motion_amplitudes[cid] <= motion_threshold:
                    reason = f"low motion amplitude ({motion_amplitudes[cid]:.1f})"
                elif points < min_points:
                    reason = f"poor continuity ({points} points < {min_points})"
                print(f"  ❌ Filtered: {cname} (ID {cid}), {reason}")
    else:
        filtered_trajectories = {}

    # 显示最终保留的轨迹
    if filtered_trajectories:
        print(f"\nFinalized {len(filtered_trajectories)} trajectories (markers)")
        total_points = sum(len(traj) for traj in filtered_trajectories.values())
        print(f"Total points: {total_points}")

    # 第三遍：可视化
    print("\nVisualizing results (press 'q' to quit)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx = 0
    detected_frames = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detected_count = 0
        for color_id, traj in filtered_trajectories.items():
            if frame_idx in traj:
                cx, cy = traj[frame_idx]
                detected_count += 1

                color_bgr = tuple(
                    next(
                        (c["bgr"] for c in colors if c["id"] == color_id),
                        [128, 128, 128],
                    )
                )

                cv2.circle(frame, (int(cx), int(cy)), 8, color_bgr, -1)
                cv2.circle(frame, (int(cx), int(cy)), 10, (255, 255, 255), 2)

                label = f"ID:{color_id}"
                cv2.putText(
                    frame,
                    label,
                    (int(cx) + 15, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

        if detected_count > 0:
            detected_frames += 1

        now = time.time()
        dt = now - last_time
        current_fps = 1.0 / dt if dt > 0 else 0.0
        last_time = now

        status = f"FPS:{current_fps:.1f} | Frame:{frame_idx}/{total_frames} | Detected:{detected_count}"
        cv2.putText(
            frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

        cv2.imshow("Color Dots Tracking", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == 27:  # q键 或 ESC键
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nExited normally")

    print("\n=== Statistics ===")
    print(f"Total frames: {frame_idx}")
    if frame_idx > 0:
        print(
            f"Frames with markers detected: {detected_frames} ({100.0 * detected_frames / frame_idx:.2f}%)"
        )
    else:
        print("Warning: No frames read")
    print("\nRetained trajectories:")
    for cid, traj in filtered_trajectories.items():
        cname = next((c["name"] for c in colors if c["id"] == cid), f"Color{cid}")
        print(f"  {cname} (ID {cid}): {len(traj)} points")


if __name__ == "__main__":
    main()
