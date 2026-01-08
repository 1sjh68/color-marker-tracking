"""
视频取色工具 - 从视频任意帧取色标定

使用方法：
  python 0_video_color_picker.py [--video path/to/video.mp4]
  不指定视频时，自动扫描videos目录供选择

功能：
1. 拖动滑动条跳转到任意帧
2. 按数字键 0-5 选择要标定的颜色
3. 点击画面中的颜色区域进行取色
4. 程序自动计算HSV范围并实时显示检测效果
5. 按 's' 保存配置到 colors.json
6. 按 'q' 退出

快捷键：
  左/右方向键: 后退/前进1帧
  上/下方向键: 前进/后退10帧
  空格: 暂停/播放
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.core.utils import (
    detect_color_with_wrap,
    load_config,
    put_chinese_text,
    get_config_path,
    get_data_dir,
)


def save_config(config):
    config_path = str(get_config_path())
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print("✓ Config saved")


def find_videos(video_dir):
    """查找所有视频文件"""
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    video_files = set()

    video_path = Path(video_dir)
    if not video_path.exists():
        return []

    for ext in video_extensions:
        video_files.update(video_path.glob(f"*{ext}"))
        video_files.update(video_path.glob(f"*{ext.upper()}"))

    return sorted(list(video_files))


def select_video_interactive():
    """交互式选择视频"""
    script_dir = Path(__file__).parent
    video_dir = get_data_dir("raw").resolve()

    print("=" * 60)
    print("Scanning video directory...")
    videos = find_videos(video_dir)

    if not videos:
        print("Error: No video files found")
        print(f"Directory: {video_dir}")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        return None

    print(f"\nFound {len(videos)} video files:")
    for i, video in enumerate(videos, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  {i}. {video.name} ({size_mb:.1f} MB)")

    while True:
        try:
            choice = input(f"\nPlease select a video (1-{len(videos)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(videos):
                selected = videos[idx]
                print(f"✓ Selected: {selected.name}")
                return str(selected)
            else:
                print(f"Please enter a number between 1 and {len(videos)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            return None


def calculate_hsv_range(hsv_sample, margin_h=12, margin_s=50, margin_v=60):
    """计算HSV范围（平衡容差以提高检测率和准确性）"""
    h, s, v = hsv_sample

    # H通道是环形的（0-180）
    h_lower = max(0, h - margin_h)
    h_upper = min(180, h + margin_h)

    # S和V通道（饱和度下限80，兼顾模糊图像和皮肤排除）
    # 注：如仍误检皮肤，使用排除模式(按E)点击皮肤手动提高下限
    s_lower = max(80, s - margin_s)
    s_upper = min(255, s + margin_s)

    v_lower = max(0, v - margin_v)
    v_upper = min(255, v + margin_v)

    return [int(h_lower), int(s_lower), int(v_lower)], [
        int(h_upper),
        int(s_upper),
        int(v_upper),
    ]


class VideoColorPicker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.config = load_config()
        self.colors = self.config["colors"]
        self.current_color_id = None
        self.sampling = False
        self.sample_point = None
        self.exclude_mode = False  # 反选排除模式
        self.window_name = "Video Color Picker - Drag to Seek"

        # 视频相关
        self.cap = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.playing = False
        self.current_frame = None
        # 防止程序设置滑动条时触发回调导致的递归和错乱
        self.suppress_trackbar_cb = False
        # 显示缩放（保持宽高比）设置
        self.view_max_width = 1280
        self.view_max_height = 720
        self.display_scale = 1.0

        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_color_id is not None and self.current_frame is not None:
                # 窗口坐标直接使用（现在显示原始尺寸，无需映射）
                h, w = self.current_frame.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    self.sample_point = (x, y)
                    self.sampling = True

    def set_frame_idx(self, idx: int, update_trackbar: bool = True) -> bool:
        """跳转到指定帧并更新当前帧缓存

        Args:
            idx: 目标帧索引
            update_trackbar: 是否同步更新滑动条位置（在trackbar回调中应为False以避免递归）
        Returns:
            bool: 是否成功读取该帧
        """
        idx = int(max(0, min(self.total_frames - 1, idx)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_idx = idx
            self.current_frame = frame.copy()
            if update_trackbar:
                self.suppress_trackbar_cb = True
                cv2.setTrackbarPos("Frame", self.window_name, self.current_frame_idx)
                self.suppress_trackbar_cb = False
        return ret

    def trackbar_callback(self, value):
        """滑动条回调：仅跳转并更新缓存，不回调设置滑动条避免递归"""
        if self.suppress_trackbar_cb:
            return
        # 用户拖动时自动暂停，避免与播放状态竞争
        self.playing = False
        self.set_frame_idx(value, update_trackbar=False)

    def sample_color(self, frame, x, y, sample_size=20):
        """在指定位置采样颜色"""
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

    def exclude_color(self, hsv_exclude, color_def):
        """反选排除：调整HSV范围以排除指定颜色

        Args:
            hsv_exclude: 要排除的HSV值 [H, S, V]
            color_def: 当前颜色定义
        """
        h_ex, s_ex, v_ex = hsv_exclude
        lower = color_def["hsv_lower"]
        upper = color_def["hsv_upper"]

        # 策略：收缩HSV范围以排除干扰值
        # 优先调整饱和度（最有效排除皮肤等低饱和度区域）
        if s_ex < lower[1]:
            # 干扰区域饱和度更低，提高下限
            lower[1] = max(lower[1], int(s_ex + 20))
            print(f"  ↑ Increase saturation lower bound: {lower[1]}")
        elif s_ex > upper[1]:
            # 干扰区域饱和度更高，降低上限
            upper[1] = min(upper[1], int(s_ex - 20))
            print(f"  ↓ Decrease saturation upper bound: {upper[1]}")
        else:
            # 干扰区域在饱和度范围内，收缩到远离干扰值的一侧
            s_mid = (lower[1] + upper[1]) // 2
            if s_ex < s_mid:
                lower[1] = min(255, int(s_ex + 15))
                print(
                    f"  ↑ Shrink saturation lower bound: {lower[1]} (exclude interference)"
                )
            else:
                upper[1] = max(0, int(s_ex - 15))
                print(
                    f"  ↓ Shrink saturation upper bound: {upper[1]} (exclude interference)"
                )

        # 调整明度（次要）
        if v_ex < lower[2]:
            lower[2] = max(lower[2], int(v_ex + 15))
            print(f"  ↑ Increase value lower bound: {lower[2]}")
        elif v_ex > upper[2]:
            upper[2] = min(upper[2], int(v_ex - 15))
            print(f"  ↓ Decrease value upper bound: {upper[2]}")

        # 色调调整（谨慎，避免过度收缩）
        if h_ex >= lower[0] and h_ex <= upper[0]:
            if abs(h_ex - lower[0]) < abs(h_ex - upper[0]):
                lower[0] = min(180, int(h_ex + 5))
                print(f"  ↑ Shrink hue lower bound: {lower[0]}")
            else:
                upper[0] = max(0, int(h_ex - 5))
                print(f"  ↓ Shrink hue upper bound: {upper[0]}")

        # 确保范围有效性
        if lower[1] > upper[1]:
            lower[1], upper[1] = upper[1], lower[1]
        if lower[2] > upper[2]:
            lower[2], upper[2] = upper[2], lower[2]

        color_def["hsv_lower"] = lower
        color_def["hsv_upper"] = upper

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

        # 增强形态学处理
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 椭圆拟合参数
        MIN_CONTOUR_POINTS = 5
        MAX_ASPECT_RATIO = 15.0
        MIN_AREA = 100
        NMS_DISTANCE = 30

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

                    if aspect_ratio <= MAX_ASPECT_RATIO:
                        candidates.append((cx, cy, area, ellipse, contour))
                except cv2.error:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                        candidates.append((cx, cy, area, None, contour))
            else:
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
                    candidates.append((cx, cy, area, None, contour))

        # NMS去重
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
        """绘制用户界面"""
        overlay = frame.copy()

        # 顶部信息栏
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 150), (50, 50, 50), -1)

        # 标题
        cv2.putText(
            overlay,
            f"Video Color Picker - Frame: {self.current_frame_idx}/{self.total_frames}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 说明
        mode_text = "[EXCLUDE]" if self.exclude_mode else "[SAMPLE]"
        help_text = f"{mode_text} 0-{max(0, len(self.colors) - 1)}: Select | E: Mode | S: Save | Space: Play | Q: Quit"
        cv2.putText(
            overlay,
            help_text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255) if self.exclude_mode else (200, 200, 200),
            2 if self.exclude_mode else 1,
        )

        # 颜色列表
        y_offset = 110
        for i, color_def in enumerate(self.colors):
            color_id = color_def["id"]
            color_name = color_def["name"]
            color_bgr = tuple(color_def["bgr"])

            # 高亮当前选中的颜色
            if color_id == self.current_color_id:
                cv2.rectangle(
                    overlay, (10, y_offset - 5), (250, y_offset + 25), (0, 255, 255), 2
                )

            # 颜色方块
            cv2.rectangle(overlay, (20, y_offset), (50, y_offset + 20), color_bgr, -1)
            cv2.rectangle(
                overlay, (20, y_offset), (50, y_offset + 20), (255, 255, 255), 1
            )

            y_offset += 30

        # 混合
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 添加中文文本
        y_offset = 110
        for i, color_def in enumerate(self.colors):
            color_id = color_def["id"]
            color_name = color_def["name"]

            # 颜色名称
            text = f"{color_id}: {color_name}"
            frame = put_chinese_text(
                frame, text, (60, y_offset), font_size=16, color=(255, 255, 255)
            )

            # HSV范围
            hsv_text = f"HSV: {color_def['hsv_lower']} - {color_def['hsv_upper']}"
            frame = put_chinese_text(
                frame, hsv_text, (200, y_offset), font_size=12, color=(180, 180, 180)
            )

            y_offset += 30

        return frame

    def run(self):
        # 转换为绝对路径
        video_path_abs = os.path.abspath(self.video_path)
        self.cap = cv2.VideoCapture(video_path_abs)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video {video_path_abs}")
            print("Please check if the file exists")
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        print("=" * 60)
        print("Video Color Picker (Supports Exclusion Mode)")
        print("=" * 60)
        print(f"Video: {os.path.basename(self.video_path)}")
        print(f"Total frames: {self.total_frames}, FPS: {fps:.2f}")
        print("=" * 60)
        print("Instructions:")
        print("  Seek Bar   - Jump to specific frame")
        print(f"  0-{len(self.colors) - 1}        - Select color to calibrate")
        print("  Click      - Click target color in window")
        print("  e          - Switch mode (Calibrate/Exclude)")
        print("               [Calibrate] Click target -> Set HSV range")
        print("               [Exclude]   Click interference -> Shrink HSV range")
        print("  ←/→       - Back/Forward 1 frame")
        print("  ↑/↓       - Forward/Back 10 frames")
        print("  Space      - Play/Pause")
        print("  s          - Save config")
        print("  q          - Quit")
        print("=" * 60)

        # 创建滑动条
        cv2.createTrackbar(
            "Frame", self.window_name, 0, self.total_frames - 1, self.trackbar_callback
        )

        # 读取第一帧
        if not self.set_frame_idx(0, update_trackbar=True):
            print("无法读取第一帧")
            return

        # 平台兼容的方向键键值集合（Windows: waitKeyEx 大值；Linux/X11: 65361-65364）
        KEY_LEFTS = {2424832, 65361}
        KEY_UPS = {2490368, 65362}
        KEY_RIGHTS = {2555904, 65363}
        KEY_DOWNS = {2621440, 65364}

        while True:
            # 播放：顺序前进；暂停：保持当前帧
            if self.playing:
                # 使用统一的跳转函数，保证索引与滑动条同步
                if not self.set_frame_idx(
                    self.current_frame_idx + 1, update_trackbar=True
                ):
                    # 到末尾则停在最后一帧并暂停
                    self.playing = False
                    self.set_frame_idx(self.total_frames - 1, update_trackbar=True)

            # 使用缓存的当前帧作为显示底图
            frame_display = self.current_frame.copy()

            # 如果正在采样
            if self.sampling and self.sample_point:
                x, y = self.sample_point

                # 绘制采样区域
                cv2.rectangle(
                    frame_display, (x - 10, y - 10), (x + 10, y + 10), (0, 255, 255), 2
                )

                # 采样颜色
                bgr_mean, hsv_mean = self.sample_color(self.current_frame, x, y)
                color_def = self.colors[self.current_color_id]

                if self.exclude_mode:
                    # 反选排除模式
                    print(
                        f"\nExcluding interference area ({color_def['name']}, frame {self.current_frame_idx}):"
                    )
                    print(f"  Interference HSV: {hsv_mean.astype(int).tolist()}")
                    self.exclude_color(hsv_mean, color_def)
                    print(
                        f"  New range: {color_def['hsv_lower']} - {color_def['hsv_upper']}"
                    )
                else:
                    # 正向标定模式
                    hsv_lower, hsv_upper = calculate_hsv_range(hsv_mean)
                    color_def["hsv_lower"] = hsv_lower
                    color_def["hsv_upper"] = hsv_upper

                    print(
                        f"\nCalibrated {color_def['name']} (frame {self.current_frame_idx}):"
                    )
                    print(f"  BGR mean: {bgr_mean.astype(int).tolist()}")
                    print(f"  HSV mean: {hsv_mean.astype(int).tolist()}")
                    print(f"  HSV range: {hsv_lower} - {hsv_upper}")

                self.sampling = False
                self.sample_point = None

            # 测试当前颜色的检测效果
            if self.current_color_id is not None:
                color_def = self.colors[self.current_color_id]
                mask, contours, ellipses = self.test_detection(
                    self.current_frame, color_def
                )

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
                h, w = frame_display.shape[:2]
                mask_w, mask_h = min(320, w // 3), min(180, h // 3)
                mask_resized = cv2.resize(mask, (mask_w, mask_h))
                mask_colored = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
                frame_display[0:mask_h, w - mask_w :] = mask_colored

            # 绘制UI
            frame_display = self.draw_ui(frame_display)

            # 直接显示原始帧，让WINDOW_NORMAL自动处理缩放（避免二次缩放导致模糊）
            self.display_scale = 1.0
            cv2.imshow(self.window_name, frame_display)

            # 按键处理（使用 waitKeyEx 以获得扩展键值，不与 0xFF 进行与运算）
            key = cv2.waitKeyEx(30 if self.playing else 1)

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
            elif key == ord(" "):
                # 空格键：播放/暂停
                self.playing = not self.playing
                print(f"{'Playing' if self.playing else 'Paused'}")
            elif key in KEY_LEFTS:  # 左箭头：后退1帧
                self.playing = False
                self.set_frame_idx(self.current_frame_idx - 1, update_trackbar=True)
            elif key in KEY_RIGHTS:  # 右箭头：前进1帧
                self.playing = False
                self.set_frame_idx(self.current_frame_idx + 1, update_trackbar=True)
            elif key in KEY_UPS:  # 上箭头：前进10帧
                self.playing = False
                self.set_frame_idx(self.current_frame_idx + 10, update_trackbar=True)
            elif key in KEY_DOWNS:  # 下箭头：后退10帧
                self.playing = False
                self.set_frame_idx(self.current_frame_idx - 10, update_trackbar=True)
            elif ord("0") <= key <= ord("9"):
                # 数字键：选择颜色通道
                color_id = key - ord("0")
                if color_id < len(self.colors):
                    self.current_color_id = color_id
                    mode_name = (
                        "Exclusion Mode" if self.exclude_mode else "Calibration Mode"
                    )
                    print(
                        f"\nSelected color: {self.colors[color_id]['name']} [{mode_name}]"
                    )
                    if self.exclude_mode:
                        print("Please click on interference area to exclude")
                    else:
                        print("Please click on target color area")

        self.cap.release()
        cv2.destroyAllWindows()
        print("\nExited")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video color picker tool")
    parser.add_argument("--video", help="Video file path (optional)")
    args = parser.parse_args()

    # 如果没有指定视频，交互式选择
    if args.video:
        video_path = args.video
    else:
        video_path = select_video_interactive()
        if not video_path:
            exit(1)

    picker = VideoColorPicker(video_path)
    picker.run()
