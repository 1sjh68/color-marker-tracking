"""
公共工具函数模块
"""

import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_project_root():
    """获取项目根目录（包含config文件夹的目录）"""
    current_dir = Path(__file__).resolve()
    # 从 src/core/ 向上找到根目录
    root = current_dir.parent.parent.parent  # src/core/ -> src/ -> 根目录
    return root


def get_config_path():
    """获取配置文件路径"""
    return get_project_root() / "config" / "colors.json"


def get_data_dir(subpath=""):
    """获取数据目录路径
    Args:
        subpath: 子路径，如 "raw", "processed/markers"
    Returns:
        Path 对象
    """
    return get_project_root() / "data" / subpath


def load_config():
    """加载颜色配置文件

    Returns:
        dict: 颜色配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        json.JSONDecodeError: JSON格式错误
        ValueError: 配置格式不正确
    """
    config_path = get_config_path()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 验证配置格式
        if "colors" not in config:
            raise ValueError("Config file missing 'colors' field")

        if not isinstance(config["colors"], list):
            raise ValueError("'colors' field must be a list")

        return config

    except FileNotFoundError:
        print(f"Error: Config file does not exist: {config_path}")
        print("Please ensure config/colors.json file exists")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Config file JSON format error: {e}")
        print(f"File path: {config_path}")
        raise
    except ValueError as e:
        print(f"Error: Config file format incorrect: {e}")
        print(f"File path: {config_path}")
        raise
    except Exception as e:
        print(f"Error: Failed to load config file: {e}")
        raise


def get_chinese_font(font_size=20):
    """跨平台获取中文字体

    Args:
        font_size: 字体大小

    Returns:
        PIL.ImageFont 对象
    """
    system = platform.system()
    font_paths = {
        "Windows": [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
        ],
        "Linux": [
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans
        ],
        "Darwin": [  # macOS
            "/System/Library/Fonts/PingFang.ttc",  # 苹方
            "/Library/Fonts/Arial Unicode.ttf",
        ],
    }

    for font_path in font_paths.get(system, []):
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, font_size)
            except (OSError, IOError):
                continue

    return ImageFont.load_default()


def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在 OpenCV 图像上绘制中文文本

    Args:
        img: OpenCV 图像（BGR 格式）
        text: 要显示的文本
        position: (x, y) 位置
        font_size: 字体大小
        color: BGR 颜色元组

    Returns:
        绘制后的图像
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = get_chinese_font(font_size)

    # 绘制文本（PIL 使用 RGB 颜色）
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

    # 转换回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def detect_color_with_wrap(hsv, lower, upper, is_red=False):
    """处理环形色调的颜色检测

    Args:
        hsv: HSV 图像
        lower: HSV 下界 [H, S, V]
        upper: HSV 上界 [H, S, V]
        is_red: 是否是红色（需要处理 H 通道跨越边界）

    Returns:
        二值掩码
    """
    if is_red:
        # 红色需要处理H通道的环形特性（0-180度）
        # 检查是否跨越0度边界：如果lower[0] > upper[0]，说明范围跨越0
        h_lower = lower[0]
        h_upper = upper[0]

        if h_lower <= h_upper:
            # 不跨界，正常处理
            lower_arr = np.array(lower, dtype=np.uint8)
            upper_arr = np.array(upper, dtype=np.uint8)
            return cv2.inRange(hsv, lower_arr, upper_arr)
        else:
            # 跨越0度边界，分成两个区间
            # 区间1: [0, h_upper]
            lower1 = np.array([0, lower[1], lower[2]], dtype=np.uint8)
            upper1 = np.array([h_upper, upper[1], upper[2]], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower1, upper1)

            # 区间2: [h_lower, 180]
            lower2 = np.array([h_lower, lower[1], lower[2]], dtype=np.uint8)
            upper2 = np.array([180, upper[1], upper[2]], dtype=np.uint8)
            mask2 = cv2.inRange(hsv, lower2, upper2)

            # 合并两个区间
            return cv2.bitwise_or(mask1, mask2)
    else:
        # 其他颜色正常处理
        lower_arr = np.array(lower, dtype=np.uint8)
        upper_arr = np.array(upper, dtype=np.uint8)
        return cv2.inRange(hsv, lower_arr, upper_arr)


def filter_trajectory(
    trajectory, min_segment_length=5, max_frame_gap=3, max_spatial_jump=50
):
    """过滤轨迹：保留所有符合条件的时间+空间连续段（允许中断）

    统一的过滤参数（平衡过滤，保留更多有效段）：
    - min_segment_length: 5 帧（最小连续段长度，降低以保留更多短段）
    - max_frame_gap: 3 帧（最大时间间隔）
    - max_spatial_jump: 50 像素（最大空间跳跃）

    注意：在翻转过程中，标记点可能被遮挡导致中断，
    因此返回所有符合条件的连续段，而不是只保留最长的。

    Args:
        trajectory: 轨迹列表 [(frame_idx, cx, cy), ...]
        min_segment_length: 最小段落长度
        max_frame_gap: 最大帧间隔
        max_spatial_jump: 最大空间跳跃距离

    Returns:
        过滤后的所有连续段合并的轨迹
    """
    if not trajectory:
        return []

    trajectory = sorted(trajectory, key=lambda x: x[0])

    segments = []
    current_segment = [trajectory[0]]

    for i in range(1, len(trajectory)):
        prev_frame, prev_cx, prev_cy = trajectory[i - 1]
        curr_frame, curr_cx, curr_cy = trajectory[i]

        time_gap = curr_frame - prev_frame
        spatial_distance = np.sqrt((curr_cx - prev_cx) ** 2 + (curr_cy - prev_cy) ** 2)

        if time_gap <= max_frame_gap and spatial_distance <= max_spatial_jump:
            current_segment.append(trajectory[i])
        else:
            if len(current_segment) >= min_segment_length:
                segments.append(current_segment)
            current_segment = [trajectory[i]]

    if len(current_segment) >= min_segment_length:
        segments.append(current_segment)

    if not segments:
        return []

    # 返回所有符合条件的段（保持分立），不合并
    # 这样可以清晰看到翻转过程中的遮挡中断
    # 返回格式：所有点的列表（已排序，但段与段之间会有断点）
    all_points = []
    for segment in segments:
        all_points.extend(segment)

    return all_points


def detect_color_ellipses(
    frame, colors, prev_positions=None, return_ellipse_params=False, roi=None
):
    """检测彩色椭圆区域（真正的椭圆拟合 + 圆度过滤）

    圆形标记在运动中会因透视变换呈现为椭圆，使用 cv2.fitEllipse 进行拟合：
    - 比 moments 更精确的中心定位
    - 可通过长宽比过滤非圆形噪声
    - 椭圆参数可用于后续姿态估计

    Args:
        frame: 当前帧
        colors: 颜色配置
        prev_positions: 上一帧各颜色的位置 {color_id: (cx, cy)}
        return_ellipse_params: 是否返回椭圆参数（长短轴、角度）
        roi: 感兴趣区域 [x1, y1, x2, y2]，None表示全帧

    Returns:
        detected: [(color_id, cx, cy), ...] 或 [(color_id, cx, cy, width, height, angle), ...]
                  每种颜色最多一个点
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = []

    # ROI掩码（如果指定了ROI，只在该区域内检测）
    h, w = frame.shape[:2]
    roi_mask = None
    if roi is not None:
        x1, y1, x2, y2 = roi
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = 255

    # 椭圆拟合参数
    MIN_CONTOUR_POINTS = 5  # fitEllipse 至少需要 5 个点
    MAX_ASPECT_RATIO = 15.0  # 最大长宽比（放宽以检测侧面视角的扁椭圆）
    MIN_AREA = 100  # 最小面积
    NMS_DISTANCE = 30  # NMS 最小中心距离（像素）

    # 椭圆质量检验参数
    MIN_CIRCULARITY = 0.3  # 最小圆度（4π×面积/周长²，椭圆≈0.7-1.0）
    MIN_AREA_RATIO = 0.5  # 轮廓面积与椭圆面积的最小比值

    def nms_candidates(candidates, min_dist=NMS_DISTANCE):
        """非极大值抑制：根据中心点距离去除重复检测"""
        if len(candidates) <= 1:
            return candidates
        # 按面积从大到小排序
        sorted_cands = sorted(candidates, key=lambda c: c[2], reverse=True)
        keep = []
        for cand in sorted_cands:
            cx, cy = cand[0], cand[1]
            is_duplicate = False
            for kept in keep:
                dist = np.sqrt((cx - kept[0]) ** 2 + (cy - kept[1]) ** 2)
                if dist < min_dist:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(cand)
        return keep

    for color_def in colors:
        color_id = color_def["id"]

        # 支持多HSV范围：优先使用 hsv_ranges，否则回退到 hsv_lower/hsv_upper
        if "hsv_ranges" in color_def and color_def["hsv_ranges"]:
            hsv_ranges = color_def["hsv_ranges"]
        else:
            # 向后兼容：将单个范围转为列表格式
            hsv_ranges = [
                {"lower": color_def["hsv_lower"], "upper": color_def["hsv_upper"]}
            ]

        # 红色需要特殊处理（ID=0）
        is_red = color_id == 0

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
        # 闭合操作：填充小孔洞，连接断开的区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        # 开操作：去除小噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

        # 应用ROI掩码（如果指定）
        if roi_mask is not None:
            mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 收集当前颜色的所有候选点
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # 过滤明显的噪声点
            if area < MIN_AREA:
                continue

            # 椭圆拟合需要至少 5 个点
            if len(contour) < MIN_CONTOUR_POINTS:
                # 轮廓点数不足，回退到矩方法
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                # 无椭圆参数时使用默认值
                candidates.append((cx, cy, area, 0, 0, 0))
                continue

            # 真正的椭圆拟合
            try:
                ellipse = cv2.fitEllipse(contour)
                (cx, cy), (width, height), angle = ellipse

                # 计算长宽比
                minor_axis = min(width, height)
                major_axis = max(width, height)
                aspect_ratio = major_axis / max(minor_axis, 1e-6)

                # 过滤极端拉伸的形状（可能是噪声或边缘）
                if aspect_ratio > MAX_ASPECT_RATIO:
                    continue

                # === 椭圆质量检验 ===

                # 1. 圆度检验：4π×面积/周长²
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < MIN_CIRCULARITY:
                        continue  # 不够圆/椭圆形，跳过

                # 2. 面积匹配度检验：轮廓面积 vs 拟合椭圆面积
                ellipse_area = np.pi * (width / 2) * (height / 2)
                if ellipse_area > 0:
                    area_ratio = area / ellipse_area
                    # 比值应接近1，太小说明轮廓是碎片，太大说明有多个连通区域
                    if area_ratio < MIN_AREA_RATIO or area_ratio > 2.0:
                        continue  # 面积不匹配，跳过

                candidates.append((cx, cy, area, width, height, angle))

            except cv2.error:
                # fitEllipse 可能因轮廓形状异常而失败，回退到矩方法
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                candidates.append((cx, cy, area, 0, 0, 0))

        # NMS：去除重叠的候选点
        candidates = nms_candidates(candidates)

        # 如果有多个候选点，选择最合理的一个
        if candidates:
            if prev_positions and color_id in prev_positions:
                # 有上一帧位置：选择距离最近的
                prev_cx, prev_cy = prev_positions[color_id]
                best_candidate = min(
                    candidates,
                    key=lambda c: np.sqrt(
                        (c[0] - prev_cx) ** 2 + (c[1] - prev_cy) ** 2
                    ),
                )
            else:
                # 没有上一帧：选择面积最大的
                best_candidate = max(candidates, key=lambda c: c[2])

            if return_ellipse_params:
                # 返回完整椭圆参数：(color_id, cx, cy, width, height, angle)
                detected.append(
                    (
                        color_id,
                        best_candidate[0],
                        best_candidate[1],
                        best_candidate[3],
                        best_candidate[4],
                        best_candidate[5],
                    )
                )
            else:
                # 兼容旧接口：只返回 (color_id, cx, cy)
                detected.append((color_id, best_candidate[0], best_candidate[1]))

    return detected
