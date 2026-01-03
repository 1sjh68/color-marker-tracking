"""
生成彩色圆点标记图片
"""

import json
import os

import cv2
import numpy as np


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "colors.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    colors = config["colors"]

    # 图片参数
    img_size = 1000
    dot_radius = 400

    # 输出目录
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output", "markers")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating color marker dots...")
    print(f"Image size: {img_size}x{img_size} pixels")
    print(f"Dot diameter: {dot_radius * 2} pixels")
    print("Recommended print size: 10cm x 10cm\n")

    for color_def in colors:
        color_id = color_def["id"]
        color_name = color_def["name"]
        color_bgr = tuple(color_def["bgr"])

        # 创建白色背景
        img = np.full((img_size, img_size, 3), (255, 255, 255), dtype=np.uint8)

        # 绘制彩色圆点
        center = (img_size // 2, img_size // 2)
        cv2.circle(img, center, dot_radius, color_bgr, -1)

        # 保存（使用imencode解决中文路径问题）
        filename = f"dot_{color_id}_{color_name}.png"
        output_path = os.path.join(output_dir, filename)

        # 使用imencode + tofile解决中文路径问题
        success, encoded_img = cv2.imencode(".png", img)
        if success:
            encoded_img.tofile(output_path)
            print(f"✓ {filename}")
        else:
            print(f"✗ Save failed: {filename}")

    print(f"\nDone! All markers generated in: {output_dir}")
    print("\nInstructions:")
    print("1. Print these images with a color printer")
    print("2. Print size: 10cm x 10cm (or larger)")
    print("3. Glossy photo paper is recommended for more vibrant colors")
    print("4. Paste on different faces of the rigid body")


if __name__ == "__main__":
    main()
