"""
实时摄像头检测 - 调试工具
按键说明：
  q - 退出
  空格 - 暂停/继续
  f - 切换过滤模式（原始/过滤后）
  1-6 - 只显示对应颜色
  0 - 显示所有颜色
"""

import argparse
import time

import cv2
import numpy as np
from utils import detect_color_with_wrap, load_config, put_chinese_text


def detect_color_ellipses(
    frame, colors, prev_positions=None, show_all_candidates=False
):
    """检测彩色椭圆区域（真正的椭圆拟合 + 圆度过滤）

    Args:
        show_all_candidates: True=显示所有候选点, False=只显示最优点
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = []
    all_candidates = {}  # 用于调试显示

    # 椭圆拟合参数
    MIN_CONTOUR_POINTS = 5  # fitEllipse 至少需要 5 个点
    MAX_ASPECT_RATIO = 15.0  # 最大长宽比
    MIN_AREA = 100  # 最小面积
    NMS_DISTANCE = 30  # NMS 最小中心距离
    MIN_CIRCULARITY = 0.3  # 最小圆度
    MIN_AREA_RATIO = 0.5   # 最小面积匹配度

    def nms_candidates(candidates, min_dist=NMS_DISTANCE):
        """非极大值抑制"""
        if len(candidates) <= 1:
            return candidates
        sorted_cands = sorted(candidates, key=lambda c: c[2], reverse=True)
        keep = []
        for cand in sorted_cands:
            cx, cy = cand[0], cand[1]
            is_duplicate = False
            for kept in keep:
                dist = np.sqrt((cx - kept[0])**2 + (cy - kept[1])**2)
                if dist < min_dist:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(cand)
        return keep

    for color_def in colors:
        color_id = color_def["id"]
        color_name = color_def["name"]
        lower = color_def["hsv_lower"]
        upper = color_def["hsv_upper"]

        # 红色需要特殊处理（ID=0）
        is_red = color_id == 0
        mask = detect_color_with_wrap(hsv, lower, upper, is_red=is_red)
        
        # 增强形态学处理
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            # 椭圆拟合（需要至少5个点）
            if len(contour) < MIN_CONTOUR_POINTS:
                continue
            
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (width, height), angle = ellipse

                # 长宽比过滤
                minor_axis = min(width, height)
                major_axis = max(width, height)
                aspect_ratio = major_axis / max(minor_axis, 1e-6)

                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue
                
                # 圆度检验
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < MIN_CIRCULARITY:
                        continue
                
                # 面积匹配度检验
                ellipse_area = np.pi * (width / 2) * (height / 2)
                if ellipse_area > 0:
                    area_ratio = area / ellipse_area
                    if area_ratio < MIN_AREA_RATIO or area_ratio > 2.0:
                        continue

                candidates.append((cx, cy, area, aspect_ratio))
            except cv2.error:
                continue  # 拟合失败直接跳过

        # NMS去重
        candidates = nms_candidates(candidates)
        all_candidates[color_id] = candidates

        # 选择最优点
        if candidates:
            if (
                prev_positions
                and color_id in prev_positions
                and not show_all_candidates
            ):
                prev_cx, prev_cy = prev_positions[color_id]
                best = min(
                    candidates,
                    key=lambda c: np.sqrt(
                        (c[0] - prev_cx) ** 2 + (c[1] - prev_cy) ** 2
                    ),
                )
            else:
                best = max(candidates, key=lambda c: c[2])

            detected.append((color_id, color_name, best[0], best[1], best[2]))

    return detected, all_candidates


def draw_detections(
    frame, detected, all_candidates, colors, show_mode, filter_color_id
):
    """绘制检测结果

    Args:
        show_mode: 'filtered' or 'all'
        filter_color_id: None=全部显示, 数字=只显示该ID
    """
    overlay = frame.copy()

    for color_def in colors:
        color_id = color_def["id"]

        # 颜色过滤
        if filter_color_id is not None and color_id != filter_color_id:
            continue

        color_bgr = tuple(color_def["bgr"])

        if show_mode == "all":
            # 显示所有候选点（半透明）
            candidates = all_candidates.get(color_id, [])
            for cx, cy, area in candidates:
                cv2.circle(overlay, (int(cx), int(cy)), 5, color_bgr, -1)
                cv2.putText(
                    overlay,
                    f"{int(area)}",
                    (int(cx) + 10, int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color_bgr,
                    1,
                )
        else:
            # 只显示最优点
            for cid, cname, cx, cy, area in detected:
                if cid == color_id:
                    cv2.circle(overlay, (int(cx), int(cy)), 10, color_bgr, -1)
                    cv2.circle(overlay, (int(cx), int(cy)), 12, (255, 255, 255), 2)

    # 混合原图和覆盖层
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # 在混合后添加中文标签（避免被透明度影响）
    if show_mode == "filtered":
        for cid, cname, cx, cy, area in detected:
            if filter_color_id is None or cid == filter_color_id:
                label = f"{cname} ({int(area)}px)"
                frame = put_chinese_text(
                    frame,
                    label,
                    (int(cx) + 15, int(cy) - 5),
                    font_size=16,
                    color=(0, 255, 255),
                )

    return frame


def main():
    parser = argparse.ArgumentParser(description="Real-time detection debug tool")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    config = load_config()
    colors = config["colors"]

    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Unable to open camera {args.camera}")
        print("Tip: Try using --camera 1 or other indices")
        cap.release()  # 释放资源
        return

    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print("Real-time Detection Debug Tool")
    print("=" * 60)
    print("Hotkeys:")
    print("  q       - Quit")
    print("  Space   - Pause/Resume")
    print("  f       - Switch mode (Filtered/All Candidates)")
    print("  1-6     - Focus on specific color ID")
    print("  0       - Show all colors")
    print("=" * 60)

    prev_positions = {}
    paused = False
    show_mode = "filtered"  # 'filtered' or 'all'
    filter_color_id = None  # None=显示所有, 数字=只显示该ID
    last_time = time.time()
    fps = 0.0  # 初始化FPS变量，避免第一帧时未定义

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Unable to read from camera")
                break

            # 检测
            detected, all_candidates = detect_color_ellipses(
                frame, colors, prev_positions, show_all_candidates=(show_mode == "all")
            )

            # 更新位置
            if show_mode == "filtered":
                prev_positions = {cid: (cx, cy) for cid, _, cx, cy, _ in detected}

            # 绘制
            frame = draw_detections(
                frame, detected, all_candidates, colors, show_mode, filter_color_id
            )

            # 计算FPS
            now = time.time()
            dt = now - last_time
            if dt > 0:
                fps = 1.0 / dt
                last_time = now
            # else: 保持上一次的 fps 值

            # 显示信息（使用中文字体）
            # Bug #1 修复：防止数组越界
            display_text = "All Colors"
            if filter_color_id is not None:
                if 0 <= filter_color_id < len(colors):
                    display_text = colors[filter_color_id]["name"]
                else:
                    display_text = f"ID {filter_color_id} (Invalid)"

            info_lines = [
                f"FPS: {fps:.1f}",
                f"Mode: {'Filtered' if show_mode == 'filtered' else 'All Candidates'}",
                f"Display: {display_text}",
                f"Detected: {len(detected)} markers",
            ]

            y_offset = 30
            for line in info_lines:
                frame = put_chinese_text(
                    frame, line, (10, y_offset), font_size=20, color=(0, 255, 0)
                )
                y_offset += 35

            # 显示颜色列表（使用中文字体）
            y_offset = frame.shape[0] - 180
            frame = put_chinese_text(
                frame, "Color List:", (10, y_offset), font_size=18, color=(255, 255, 255)
            )
            y_offset += 30

            for color_def in colors:
                cid = color_def["id"]
                cname = color_def["name"]
                count = len(all_candidates.get(cid, []))
                text = f"{cid}: {cname} ({count})"
                color_bgr = tuple(color_def["bgr"])
                frame = put_chinese_text(
                    frame, text, (10, y_offset), font_size=16, color=color_bgr
                )
                y_offset += 25

        cv2.imshow("实时检测调试", frame)

        # 按键处理（增加延迟提高响应可靠性）
        key = cv2.waitKey(10) & 0xFF

        if key == ord("q") or key == 27:  # q键 或 ESC键
            break
        elif key == ord(" "):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord("f"):
            show_mode = "all" if show_mode == "filtered" else "filtered"
            print(
                f"Switched to: {'All Candidates' if show_mode == 'all' else 'Optimal Only'}"
            )
        elif key == ord("0"):
            filter_color_id = None
            print("Showing all colors")
        elif ord("1") <= key <= ord("9"):
            cid = key - ord("0")
            if cid < len(colors):
                filter_color_id = cid
                print(f"Showing: {colors[cid]['name']}")

    cap.release()
    cv2.destroyAllWindows()
    print("Exited")


if __name__ == "__main__":
    main()
