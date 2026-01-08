"""
颜色标定工具 - 从摄像头取色自动计算HSV范围

使用方法：
  python 0_color_calibration.py [--camera 0]

原始使用方法：
1. 运行程序，摄像头画面会显示
2. 按数字键 0-5 选择要标定的颜色
3. 在画面中点击该颜色的区域（会取周围20x20像素平均值）
4. 程序自动计算HSV范围并实时显示检测效果
5. 按 's' 保存配置
6. 按 'q' 退出
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.utils import (
    detect_color_with_wrap,
    load_config,
    put_chinese_text,
    get_config_path,
)


def save_config(config):
    config_path = str(get_config_path())
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("✓ Config saved")


def calculate_hsv_range(hsv_sample, margin_h=12, margin_s=60, margin_v=80):
    """计算HSV范围（无硬性限制，适应强光/过曝环境）

    Args:
        hsv_sample: 采样区域的HSV平均值 [H, S, V]
        margin_h: H通道容差（12以适应色偏）
        margin_s: S通道容差（60以适应饱和度变化，包括过曝）
        margin_v: V通道容差（80以适应明暗变化，包括强光）
    """
    h, s, v = hsv_sample

    # H通道是环形的（0-180）
    h_lower = max(0, h - margin_h)
    h_upper = min(180, h + margin_h)

    # S和V通道（不再设置硬性下限，完全根据采样值计算）
    # 强光下S值可能很低，需要允许检测
    s_lower = max(0, s - margin_s)
    s_upper = min(255, s + margin_s)

    v_lower = max(0, v - margin_v)
    v_upper = min(255, v + margin_v)

    return [int(h_lower), int(s_lower), int(v_lower)], [
        int(h_upper),
        int(s_upper),
        int(v_upper),
    ]


class ColorCalibrator:
    def __init__(self):
        self.config = load_config()
        self.colors = self.config["colors"]
        self.current_color_id = None
        self.sampling = False
        self.sample_point = None
        self.exclude_mode = False  # 反选排除模式
        self.append_mode = False  # 追加模式：True=追加新范围，False=覆盖
        self.window_name = "Color Calibration Tool - Click to Calibrate"

        # ROI框选相关
        self.roi_mode = False  # ROI选择模式
        self.roi_selecting = False  # 正在框选
        self.roi_start = None  # 框选起点
        self.roi_end = None  # 框选终点
        self.roi = self.config.get("roi", None)  # 从配置加载ROI [x1, y1, x2, y2]

        # 创建窗口和鼠标回调
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # 显示缩放（保持宽高比）设置
        self.view_max_width = 1280
        self.view_max_height = 720
        self.display_scale = 1.0

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调：支持点击采样和拖拽框选ROI"""
        # 将显示坐标映射回源图坐标
        sx = 1.0 / max(self.display_scale, 1e-6)
        src_x = int(round(x * sx))
        src_y = int(round(y * sx))

        if self.roi_mode:
            # ROI框选模式
            if event == cv2.EVENT_LBUTTONDOWN:
                self.roi_selecting = True
                self.roi_start = (src_x, src_y)
                self.roi_end = (src_x, src_y)
            elif event == cv2.EVENT_MOUSEMOVE and self.roi_selecting:
                self.roi_end = (src_x, src_y)
            elif event == cv2.EVENT_LBUTTONUP and self.roi_selecting:
                self.roi_selecting = False
                self.roi_end = (src_x, src_y)
                # 确保坐标顺序正确
                x1 = min(self.roi_start[0], self.roi_end[0])
                y1 = min(self.roi_start[1], self.roi_end[1])
                x2 = max(self.roi_start[0], self.roi_end[0])
                y2 = max(self.roi_start[1], self.roi_end[1])
                if x2 - x1 > 10 and y2 - y1 > 10:  # 最小尺寸
                    self.roi = [x1, y1, x2, y2]
                    self.config["roi"] = self.roi
                    print(f"\n✓ ROI set: [{x1}, {y1}] - [{x2}, {y2}]")
                    print(f"  Size: {x2 - x1} x {y2 - y1} pixels")
                else:
                    print("\n✗ ROI too small, ignored")
                self.roi_mode = False
                print("ROI mode OFF")
        else:
            # 普通点击采样模式
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.current_color_id is not None:
                    self.sample_point = (src_x, src_y)
                    self.sampling = True

    def sample_color(self, frame, x, y, sample_size=20):
        """在指定位置采样颜色

        Args:
            frame: BGR图像
            x, y: 采样中心点
            sample_size: 采样区域大小

        Returns:
            bgr_mean, hsv_mean: BGR和HSV平均值
        """
        h, w = frame.shape[:2]

        # 确保采样区域在图像内
        x1 = max(0, x - sample_size // 2)
        y1 = max(0, y - sample_size // 2)
        x2 = min(w, x + sample_size // 2)
        y2 = min(h, y + sample_size // 2)

        # 采样区域
        sample_bgr = frame[y1:y2, x1:x2]
        sample_hsv = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2HSV)

        # 计算平均值
        bgr_mean = np.mean(sample_bgr, axis=(0, 1))
        hsv_mean = np.mean(sample_hsv, axis=(0, 1))

        return bgr_mean, hsv_mean

    def set_hsv_range(self, color_def, hsv_lower, hsv_upper):
        """设置HSV范围（支持覆盖和追加模式）

        Args:
            color_def: 颜色定义
            hsv_lower: HSV下界
            hsv_upper: HSV上界
        """
        new_range = {"lower": hsv_lower, "upper": hsv_upper}

        if self.append_mode:
            # 追加模式：添加到 hsv_ranges 数组
            if "hsv_ranges" not in color_def:
                # 如果还没有 hsv_ranges，先用现有范围初始化
                color_def["hsv_ranges"] = [
                    {"lower": color_def["hsv_lower"], "upper": color_def["hsv_upper"]}
                ]
            color_def["hsv_ranges"].append(new_range)
            print(f"  [APPEND] Total ranges: {len(color_def['hsv_ranges'])}")
        else:
            # 覆盖模式：替换单一范围，清除 hsv_ranges
            color_def["hsv_lower"] = hsv_lower
            color_def["hsv_upper"] = hsv_upper
            if "hsv_ranges" in color_def:
                del color_def["hsv_ranges"]
            print(f"  [OVERWRITE] Single range set")

    def exclude_color(self, hsv_exclude, color_def):
        """反选排除：添加排除区域到hsv_excludes列表

        排除逻辑：在并集结果中挖掉指定的HSV区域
        检测掩码 = (范围1 ∪ 范围2 ∪ ...) - (排除1 ∪ 排除2 ∪ ...)

        Args:
            hsv_exclude: 要排除的HSV值 [H, S, V]
            color_def: 当前颜色定义
        """
        h_ex, s_ex, v_ex = [int(x) for x in hsv_exclude]

        # 计算排除范围（以采样点为中心的小区域）
        margin_h = 10
        margin_s = 40
        margin_v = 50

        exclude_lower = [
            max(0, h_ex - margin_h),
            max(0, s_ex - margin_s),
            max(0, v_ex - margin_v),
        ]
        exclude_upper = [
            min(180, h_ex + margin_h),
            min(255, s_ex + margin_s),
            min(255, v_ex + margin_v),
        ]

        # 初始化排除列表（如果不存在）
        if "hsv_excludes" not in color_def:
            color_def["hsv_excludes"] = []

        # 添加排除范围
        color_def["hsv_excludes"].append(
            {"lower": exclude_lower, "upper": exclude_upper}
        )

        print(
            f"  + Added exclusion zone: H[{exclude_lower[0]}-{exclude_upper[0]}] S[{exclude_lower[1]}-{exclude_upper[1]}] V[{exclude_lower[2]}-{exclude_upper[2]}]"
        )
        print(f"  Total exclusion zones: {len(color_def['hsv_excludes'])}")

    def test_detection(self, frame, color_def):
        """测试检测效果（使用椭圆拟合+NMS，支持多范围）"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 支持多HSV范围：优先使用 hsv_ranges，否则回退到 hsv_lower/hsv_upper
        if "hsv_ranges" in color_def and color_def["hsv_ranges"]:
            hsv_ranges = color_def["hsv_ranges"]
        else:
            hsv_ranges = [
                {"lower": color_def["hsv_lower"], "upper": color_def["hsv_upper"]}
            ]

        # 红色需要特殊处理
        is_red = color_def["id"] == 0

        # 对每个HSV范围检测并合并掩码
        combined_mask = None
        for hsv_range in hsv_ranges:
            lower = hsv_range["lower"]
            upper = hsv_range["upper"]
            range_mask = detect_color_with_wrap(hsv, lower, upper, is_red=is_red)
            if combined_mask is None:
                combined_mask = range_mask
            else:
                combined_mask = cv2.bitwise_or(combined_mask, range_mask)

        mask = combined_mask

        # 应用排除区域（从并集中挖掉）
        if "hsv_excludes" in color_def and color_def["hsv_excludes"]:
            for exclude_range in color_def["hsv_excludes"]:
                ex_lower = np.array(exclude_range["lower"], dtype=np.uint8)
                ex_upper = np.array(exclude_range["upper"], dtype=np.uint8)
                exclude_mask = cv2.inRange(hsv, ex_lower, ex_upper)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

        # 增强形态学处理：先闭合（连接断点），再开操作（去噪）
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 应用ROI掩码（如果指定）
        if self.roi:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = self.roi
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[y1:y2, x1:x2] = 255
            mask = cv2.bitwise_and(mask, roi_mask)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 椭圆拟合参数
        MIN_CONTOUR_POINTS = 5
        MAX_ASPECT_RATIO = 15.0
        MIN_AREA = 100
        NMS_DISTANCE = 30
        MIN_CIRCULARITY = 0.3  # 最小圆度
        MIN_AREA_RATIO = 0.5  # 最小面积匹配度

        # 收集有效的椭圆
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < MIN_AREA:
                continue

            if len(contour) >= MIN_CONTOUR_POINTS:
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

                    candidates.append((cx, cy, area, ellipse, contour))
                except cv2.error:
                    pass  # 拟合失败直接跳过，不回退
            # 点数不足也跳过，不使用矩方法回退

        # NMS：去除重叠的候选点
        if len(candidates) > 1:
            sorted_cands = sorted(candidates, key=lambda c: c[2], reverse=True)
            keep = []
            for cand in sorted_cands:
                cx, cy = cand[0], cand[1]
                is_duplicate = False
                for kept in keep:
                    dist = np.sqrt((cx - kept[0]) ** 2 + (cy - kept[1]) ** 2)
                    if dist < NMS_DISTANCE:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    keep.append(cand)
            candidates = keep

        # 转换为原始格式返回
        ellipses = [(c[3], c[4]) for c in candidates]

        return mask, contours, ellipses

    def draw_ui(self, frame):
        """绘制用户界面（水平布局）"""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # 顶部信息栏（更紧凑）
        bar_height = 70
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (40, 40, 40), -1)

        # 标题 + 模式
        if self.roi_mode:
            title_text = "[ROI SELECT] Drag to select region"
            title_color = (0, 255, 255)
        else:
            mode_text = "[EXCLUDE]" if self.exclude_mode else "[SAMPLE]"
            append_text = "[+APPEND]" if self.append_mode else ""
            roi_text = "[ROI]" if self.roi else ""
            title_text = f"Calibration {mode_text} {append_text} {roi_text}"
            title_color = (
                (0, 255, 0)
                if self.append_mode
                else ((0, 255, 255) if self.exclude_mode else (255, 255, 255))
            )
        cv2.putText(
            overlay,
            title_text,
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            title_color,
            2,
        )

        # 快捷键说明
        help_text = "0-5:Sel | E:Mode | A:Append | O:ROI | R:ClrROI | S:Save | Q:Quit"
        cv2.putText(
            overlay,
            help_text,
            (15, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        # 水平颜色列表
        x_offset = 350
        item_width = 140
        for i, color_def in enumerate(self.colors):
            color_id = color_def["id"]
            color_name = color_def["name"]
            color_bgr = tuple(color_def["bgr"])

            # 高亮当前选中
            if color_id == self.current_color_id:
                cv2.rectangle(
                    overlay,
                    (x_offset - 5, 8),
                    (x_offset + item_width - 10, bar_height - 8),
                    (0, 255, 255),
                    2,
                )

            # 颜色方块
            cv2.rectangle(overlay, (x_offset, 15), (x_offset + 25, 40), color_bgr, -1)
            cv2.rectangle(
                overlay, (x_offset, 15), (x_offset + 25, 40), (255, 255, 255), 1
            )

            # 颜色名称
            cv2.putText(
                overlay,
                f"{color_id}:{color_name}",
                (x_offset + 30, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )

            # HSV范围（简化显示）
            hsv_short = f"H:{color_def['hsv_lower'][0]}-{color_def['hsv_upper'][0]}"
            cv2.putText(
                overlay,
                hsv_short,
                (x_offset + 5, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (150, 150, 150),
                1,
            )

            x_offset += item_width

        # 混合
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        return frame

    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Unable to open camera {camera_id}")
            print("Tip: Try using --camera 1 or other indices")
            cap.release()  # 释放资源
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 60)
        print("Color Calibration Tool (Multi-Range Support)")
        print("=" * 60)
        print("Operation Instructions:")
        print("  0-5  - Select color to calibrate")
        print("  Click- Click on the target color area in the window")
        print("  e    - Switch mode (Calibrate/Exclude)")
        print("         [Calibrate] Click target color -> Add HSV range")
        print("         [Exclude]   Click interference -> Add exclusion zone")
        print("  a    - Toggle Append Mode (multi-range per color)")
        print("  c    - Clear all hsv_ranges for current color")
        print("  x    - Clear all exclusion zones for current color")
        print("  o    - Enter ROI selection mode (drag to select region)")
        print("  r    - Clear ROI (use full frame)")
        print("  s    - Save config")
        print("  q    - Quit")
        print("=" * 60)
        print("=" * 60)
        if self.roi:
            print(f"Loaded ROI: {self.roi}")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Unable to read from camera")
                break

            frame_display = frame.copy()

            # 如果正在采样
            if self.sampling and self.sample_point:
                x, y = self.sample_point

                # 采样颜色
                bgr_mean, hsv_mean = self.sample_color(frame, x, y)
                color_def = self.colors[self.current_color_id]

                if self.exclude_mode:
                    # 反选排除模式
                    print(f"\nExcluding interference area ({color_def['name']}):")
                    print(f"  Interference HSV: {hsv_mean.astype(int).tolist()}")
                    self.exclude_color(hsv_mean, color_def)
                    print(
                        f"  New range: {color_def['hsv_lower']} - {color_def['hsv_upper']}"
                    )
                else:
                    # 正向标定模式
                    hsv_lower, hsv_upper = calculate_hsv_range(hsv_mean)

                    print(f"\nCalibrated {color_def['name']}:")
                    print(f"  BGR mean: {bgr_mean.astype(int).tolist()}")
                    print(f"  HSV mean: {hsv_mean.astype(int).tolist()}")
                    print(f"  HSV range: {hsv_lower} - {hsv_upper}")

                    # 使用新方法设置范围（支持覆盖/追加模式）
                    self.set_hsv_range(color_def, hsv_lower, hsv_upper)

                self.sampling = False
                self.sample_point = None

            # 测试当前颜色的检测效果
            if self.current_color_id is not None:
                color_def = self.colors[self.current_color_id]
                mask, contours, ellipses = self.test_detection(frame, color_def)

                # 在原图上绘制检测结果
                for ellipse_data, contour in ellipses:
                    if ellipse_data is not None:
                        # 绘制拟合的椭圆
                        cv2.ellipse(frame_display, ellipse_data, (0, 255, 0), 2)
                        # 绘制椭圆中心
                        cx, cy = int(ellipse_data[0][0]), int(ellipse_data[0][1])
                        cv2.circle(frame_display, (cx, cy), 5, (0, 0, 255), -1)
                    else:
                        # 无法拟合椭圆时绘制轮廓
                        cv2.drawContours(frame_display, [contour], -1, (0, 255, 0), 2)
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(frame_display, (cx, cy), 5, (0, 0, 255), -1)

                # 显示mask（小窗口）
                mask_resized = cv2.resize(mask, (320, 180))
                mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                frame_display[0:180, frame_display.shape[1] - 320 :] = mask_colored

            # 绘制ROI边框
            if self.roi:
                x1, y1, x2, y2 = self.roi
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(
                    frame_display,
                    "ROI",
                    (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2,
                )

            # 绘制正在框选的ROI
            if self.roi_selecting and self.roi_start and self.roi_end:
                cv2.rectangle(
                    frame_display, self.roi_start, self.roi_end, (0, 255, 255), 2
                )
                cv2.putText(
                    frame_display,
                    "Selecting ROI...",
                    (self.roi_start[0], self.roi_start[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            # 绘制UI
            frame_display = self.draw_ui(frame_display)

            # 等比缩放后显示（保持原始长宽比，完整显示）
            h, w = frame_display.shape[:2]
            scale = min(self.view_max_width / w, self.view_max_height / h)
            if scale <= 0:
                scale = 1.0
            self.display_scale = float(scale)
            if abs(scale - 1.0) < 1e-3:
                show_frame = frame_display
            else:
                show_frame = cv2.resize(
                    frame_display,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(self.window_name, show_frame)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("s"):
                save_config(self.config)
            elif key == ord("e"):
                self.exclude_mode = not self.exclude_mode
                mode_name = (
                    "Exclusion Mode" if self.exclude_mode else "Calibration Mode"
                )
                print(f"\nSwitched to: {mode_name}")
                if self.exclude_mode:
                    print(
                        "Tip: Click on unwanted areas (e.g. skin) to shrink HSV range"
                    )
                else:
                    print("Tip: Click on target color area to calibrate")
            elif key == ord("a"):
                # 切换追加模式
                self.append_mode = not self.append_mode
                mode_name = (
                    "APPEND (Multi-Range)" if self.append_mode else "OVERWRITE (Single)"
                )
                print(f"\nAppend Mode: {mode_name}")
                if self.append_mode:
                    print("Tip: Each click will ADD a new HSV range to the color")
                else:
                    print("Tip: Each click will REPLACE the HSV range")
            elif key == ord("c"):
                # 清除当前颜色的所有额外范围
                if self.current_color_id is not None:
                    color_def = self.colors[self.current_color_id]
                    if "hsv_ranges" in color_def:
                        del color_def["hsv_ranges"]
                        print(
                            f"\nCleared all additional HSV ranges for {color_def['name']}"
                        )
                        print(
                            f"  Keeping single range: {color_def['hsv_lower']} - {color_def['hsv_upper']}"
                        )
                    else:
                        print(
                            f"\n{color_def['name']} has no additional ranges to clear"
                        )
            elif key == ord("x"):
                # 清除当前颜色的所有排除区域
                if self.current_color_id is not None:
                    color_def = self.colors[self.current_color_id]
                    if "hsv_excludes" in color_def and color_def["hsv_excludes"]:
                        count = len(color_def["hsv_excludes"])
                        del color_def["hsv_excludes"]
                        print(
                            f"\n✓ Cleared {count} exclusion zones for {color_def['name']}"
                        )
                    else:
                        print(f"\n{color_def['name']} has no exclusion zones to clear")
            elif key == ord("o"):
                # 进入ROI选择模式
                self.roi_mode = True
                print("\n[ROI MODE] Drag to select region of interest")
                print("  Detection will only work inside selected region")
            elif key == ord("r"):
                # 清除ROI
                if self.roi:
                    self.roi = None
                    if "roi" in self.config:
                        del self.config["roi"]
                    print("\n✓ ROI cleared, using full frame")
                else:
                    print("\nNo ROI to clear")
            elif ord("0") <= key <= ord("9"):
                color_id = key - ord("0")
                if color_id < len(self.colors):
                    self.current_color_id = color_id
                    color_def = self.colors[color_id]
                    mode_name = (
                        "Exclusion Mode" if self.exclude_mode else "Calibration Mode"
                    )
                    append_info = " [APPEND]" if self.append_mode else ""
                    print(
                        f"\nSelected color: {color_def['name']} [{mode_name}]{append_info}"
                    )
                    # 显示当前范围数量
                    if "hsv_ranges" in color_def:
                        print(f"  Current HSV ranges: {len(color_def['hsv_ranges'])}")
                    else:
                        print(
                            f"  Current HSV range: {color_def['hsv_lower']} - {color_def['hsv_upper']}"
                        )
                    if self.exclude_mode:
                        print("Please click on interference area to exclude")
                    else:
                        print("Please click on target color area")

        cap.release()
        cv2.destroyAllWindows()
        print("\nExited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color calibration tool")
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default 0)"
    )
    args = parser.parse_args()

    calibrator = ColorCalibrator()
    calibrator.run(camera_id=args.camera)
