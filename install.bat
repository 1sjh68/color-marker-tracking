@echo off
chcp 65001 >nul
echo ========================================
echo 彩色圆点标记跟踪系统 - 依赖安装
echo ========================================
echo.

echo 正在安装依赖...
pip install -r requirements.txt

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 快速开始：
echo   1. 颜色标定: python src\tools\0_color_calibration.py
echo   2. 生成标记: python src\core\1_generate_markers.py
echo   3. 检测跟踪: python src\core\2_detect_and_track.py --video data\raw\你的视频.mp4
echo.
pause
