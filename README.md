# 彩色标记点跟踪系统

用于贾尼别科夫效应研究的视觉追踪系统，通过彩色圆点标记实现刚体姿态跟踪。

## 核心功能

- **标记生成** - 自动生成多色彩色圆点标记图片
- **智能检测** - 基于 HSV 色彩空间的高精度检测
- **轨迹过滤** - 时空连续性过滤，自动去除背景干扰
- **运动分析** - 根据运动幅度和连续性筛选真实标记点
- **批量处理** - 支持多视频批量分析
- **数据导出** - CSV 格式导出，便于后续处理

## 快速开始

### 环境安装

```bash
# 运行安装脚本
install.bat

# 或手动安装
pip install -r requirements.txt
```

### 完整工作流

```bash
# 1. 生成标记点图片
python src/core/1_generate_markers.py

# 2. 打印标记点并贴在刚体上（推荐 10cm×10cm，光面相纸）

# 3. 拍摄视频（60fps 或更高），将视频放入 data/raw/

# 4. 检测与跟踪
python src/core/2_detect_and_track.py --video "data/raw/your_video.mp4"

# 5. 导出数据（可选）
python src/core/4_export_csv.py --video "data/raw/your_video.mp4"

# 6. 3D 可视化（可选）
python src/core/3_visualize_3d.py --video "data/raw/your_video.mp4"

# 7. 渲染检测视频（可选）
python src/core/5_render_detection_video.py --video "data/raw/your_video.mp4"
```

## 项目结构

```
视觉方案/
├── src/
│   ├── core/                   # 核心功能脚本
│   │   ├── 1_generate_markers.py    # 生成标记点图片
│   │   ├── 2_detect_and_track.py    # 检测与跟踪（主程序）
│   │   ├── 3_visualize_3d.py        # 3D 轨迹可视化
│   │   ├── 4_export_csv.py          # 导出 CSV 数据
│   │   ├── 5_render_detection_video.py  # 渲染检测视频
│   │   └── utils.py                 # 工具函数库
│   └── tools/                  # 调试和校准工具
│       ├── 0_color_calibration.py   # 颜色校准工具
│       ├── 0_realtime_debug.py      # 实时调试工具
│       └── 0_video_color_picker.py  # 视频颜色拾取工具
├── scripts/
│   ├── batch_process_videos.py  # 批量处理视频
│   ├── 6_postprocess_trajectory.py    # 轨迹后处理
│   └── 7_prepare_paper_assets_12_16.py  # 准备论文素材
├── config/
│   └── colors.json              # 颜色定义和 HSV 参数
├── data/
│   ├── raw/                    # 原始视频
│   └── processed/               # 处理后的数据
│       ├── markers/             # 生成的标记点图片
│       ├── trajectories/        # CSV 轨迹数据
│       ├── visualizations/      # 3D 可视化图片
│       └── videos/              # 渲染的检测视频
├── docs/                       # 文档
│   ├── images/                 # 文档图片
│   ├── latex/                  # LaTeX 论文文件
│   └── references/             # 参考资料和笔记
├── assets/                     # 演示文稿等资源
│   └── PPT.pdf                 # 演示文稿
├── reports/                    # 论文和报告
│   └── optics_term_paper.md    # 光学课程论文
├── requirements.txt            # Python 依赖
└── README.md                   # 本文件
```

## 脚本功能详解

### 核心脚本

#### `1_generate_markers.py` - 生成标记点

生成彩色圆点标记图片，用于打印和贴附。

```bash
python src/core/1_generate_markers.py
```

**输出**: `data/processed/markers/` 目录下的彩色圆点图片（Red, Green, Blue, Yellow）

**打印建议**:
- 尺寸: 10cm × 10cm（可根据刚体大小调整）
- 纸张: 光面相纸（颜色鲜艳、对比度高）
- 贴附: 分散在刚体不同面，确保不共面

---

#### `2_detect_and_track.py` - 检测与跟踪

核心功能：检测彩色标记点并跟踪其运动轨迹。

```bash
python src/core/2_detect_and_track.py --video "data/raw/video.mp4"
```

**算法流程**:
1. **第一遍扫描** - 收集所有帧的颜色检测结果
2. **轨迹过滤** - 应用时空连续性过滤，去除背景噪声
3. **运动分析** - 计算运动幅度，保留运动显著的轨迹
4. **可视化输出** - 实时显示跟踪结果

**过滤机制**:
- **时间连续性**: 允许最大帧间隔 ≤ 3 帧
- **空间连续性**: 相邻帧距离 ≤ 50 像素
- **最小连续段长度**: ≥ 5 帧
- **运动幅度阈值**: 位置标准差 > 5.0 像素
- **保留点数阈值**: ≥ 50 点

**输出统计**:
```
过滤轨迹...
  Red (ID 0): 562 -> 133 点, 运动幅度: 45.3
  Green (ID 1): 241 -> 0 点, 运动幅度: 2.1
  Blue (ID 2): 999 -> 20 点, 运动幅度: 8.7

过滤阈值: 运动幅度 > 5.0, 最少点数 >= 50

过滤结果:
  ✅ 保留: Red (ID 0), 点数: 133, 运动幅度: 45.3, 覆盖率: 26.2%
  ❌ 过滤: Green (ID 1), 运动幅度低 (2.1)
  ❌ 过滤: Blue (ID 2), 连续性差 (20点 < 50)
```

---

#### `4_export_csv.py` - 导出数据

将轨迹数据导出为 CSV 格式，便于后续分析。

```bash
python src/core/4_export_csv.py --video "data/raw/video.mp4" --output "data/processed/trajectories/output.csv"
```

**参数**:
- `--video`: 视频文件路径（必需）
- `--output`: CSV 输出路径（可选，默认 `data/processed/trajectories/trajectory_<视频名>.csv`）
- `--color-id`: 指定导出的颜色 ID（可选，默认导出全部颜色）

**CSV 格式**:
```csv
frame_idx,color_id,u,v
1,0,320.5,240.2
1,1,450.3,260.8
2,0,322.1,241.5
```

---

#### `3_visualize_3d.py` - 3D 可视化

生成轨迹的 3D 时空图，用于直观分析运动模式。

```bash
python src/core/3_visualize_3d.py --video "data/raw/video.mp4"
```

**输出**: 
- `data/processed/visualizations/trajectory_3d_<视频名>.png`
- 交互式 3D 图窗口

**图表说明**:
- **X 轴**: 时间（帧号）
- **Y 轴**: 像素 u 坐标
- **Z 轴**: 像素 v 坐标
- **平滑曲线** = 真实标记点
- **散乱点** = 背景噪声

---

#### `5_render_detection_video.py` - 渲染检测视频

将检测结果渲染为可视化视频。

```bash
python src/core/5_render_detection_video.py --video "data/raw/video.mp4" --filter-motion
```

**参数**:
- `--video`: 视频文件路径（必需）
- `--filter-motion`: 启用运动幅度过滤（推荐）
- `--show-trail`: 显示最近 30 帧轨迹尾迹（可选）
- `--output`: 输出视频路径（可选，默认 `data/processed/videos/detection_<视频名>.mp4`）

**输出**: `data/processed/videos/` 目录下的检测视频

---

#### `batch_process_videos.py` - 批量处理

批量处理多个视频文件。

```bash
python scripts/batch_process_videos.py --video-dir "data/raw"
```

**参数**:
- `--video-dir`: 视频目录（默认 `data/raw`）
- `--export-csv`: 导出 CSV（默认开启）
- `--visualize-3d`: 生成 3D 图（默认开启）
- `--render-video`: 渲染视频（默认开启）
- `--skip-interactive`: 跳过交互确认

**功能**: 自动扫描目录，依次处理所有视频文件

---

### 辅助工具

#### `0_color_calibration.py` - 颜色校准

交互式工具，用于校准 HSV 颜色范围。

```bash
python src/tools/0_color_calibration.py
```

**功能**:
1. 实时预览摄像头画面并点击采样
2. 支持排除模式（E）收缩 HSV 范围，减少误检
3. 支持追加模式（A）为同一颜色添加多段 HSV 范围
4. 支持 ROI 框选（O）限制检测区域，减少背景干扰
5. 按 S 保存到 `config/colors.json`

---

#### `0_video_color_picker.py` - 视频颜色拾取

从视频中拾取颜色并生成 HSV 参数。

```bash
python src/tools/0_video_color_picker.py --video "data/raw/video.mp4"
```

**功能**:
- 支持不传 `--video` 时从 `data/raw/` 目录交互式选择
- 在视频帧上点击标记点并自动推荐 HSV 范围
- 支持排除模式（E）收缩范围，减少误检
- 方向键/滑动条快速跳帧

---

#### `0_realtime_debug.py` - 实时调试

实时调试检测效果（使用摄像头）。

```bash
python src/tools/0_realtime_debug.py
```

**功能**: 实时显示检测结果，快速验证参数设置

---

## 核心技术

### 检测算法

1. **颜色空间转换**: RGB → HSV，增强色彩鲁棒性
2. **阈值分割**: 基于 HSV 范围的二值化（红色特殊处理 H 通道环形边界）
3. **形态学处理**: 开运算（去噪）+ 闭运算（填充）
4. **轮廓检测**: `cv2.findContours` 提取连通区域
5. **椭圆拟合**: `cv2.fitEllipse` 拟合椭圆（需要 ≥5 个轮廓点）
6. **圆度过滤**: 过滤长宽比 > 5.0 的极端拉伸形状
7. **质心提取**: 椭圆中心作为标记点坐标（亚像素级精度）

**椭圆拟合的优势**：
- 圆形标记在运动中因透视变换呈现为椭圆
- `fitEllipse` 比矩方法（moments）定位更精确
- 椭圆参数（长短轴、旋转角度）可用于后续姿态估计
- 通过 `return_ellipse_params=True` 可获取完整椭圆信息

### 降噪策略

**多层过滤机制**:
- **时间连续性**: 最大帧间隔 ≤ 3 帧
- **空间连续性**: 相邻帧距离 ≤ 50 像素
- **轨迹分段**: 将中断的轨迹分割为多段
- **最小长度**: 过滤长度 < 5 帧的短段
- **运动幅度**: 保留运动幅度 > 5.0 的轨迹

**智能选择**:
- 自动选择运动幅度最大且连续性好的轨迹
- 过滤背景中的静止物体和低质量轨迹

### 性能指标

| 指标 | 典型值 |
|------|--------|
| 检测率 | 95%+ |
| 降噪率 | 90%+ |
| 定位精度 | 亚像素级（椭圆拟合）|
| 处理速度 | ~200 fps（CPU）|

---

## 颜色配置

编辑 `config/colors.json` 调整颜色参数：

```json
{
  "colors": [
    {
      "id": 0,
      "name": "Red",
      "bgr": [0, 0, 255],
      "hsv_lower": [0, 132, 51],
      "hsv_upper": [7, 228, 171],
      "hsv_ranges": [
        { "lower": [0, 120, 50], "upper": [7, 228, 171] },
        { "lower": [170, 120, 50], "upper": [180, 255, 200] }
      ]
    }
  ],
  "roi": [100, 60, 1180, 660]
}
```

**参数说明**:
- `id`: 颜色唯一标识符
- `name`: 颜色名称
- `bgr`: BGR 格式的颜色值（用于绘制）
- `hsv_lower`: HSV 下界 [H, S, V]
- `hsv_upper`: HSV 上界 [H, S, V]
- `hsv_ranges`: 可选，多段 HSV 范围（用于追加/多场景）
- `roi`: 可选，ROI 区域 [x1, y1, x2, y2]

**HSV 范围调整建议**:
- **H (色调)**: 0-179，定义颜色类型
- **S (饱和度)**: 0-255，降低下界可识别更淡的颜色
- **V (亮度)**: 0-255，降低下界可识别更暗的颜色

**调优流程**:
1. 使用 `src/tools/0_video_color_picker.py` 获取推荐参数
2. 使用 `src/tools/0_color_calibration.py` 追加/排除细调并设置 ROI（可选）
3. 使用 `src/core/2_detect_and_track.py` 验证效果

---

## 拍摄建议

### 设备要求
- **推荐**: 智能手机或运动相机
- **帧率**: 60fps 或更高（240fps 更佳）
- **分辨率**: 1080p 或更高

### 拍摄技巧
- **光照**: 充足且均匀（可用台灯补光）
- **背景**: 避免与标记点颜色相似的背景
- **距离**: 保持适当距离，标记点清晰可见
- **角度**: 至少保证 1 个标记点在画面内
- **稳定性**: 尽量保持相机稳定（可用三脚架）

### 注意事项
- ❌ 避免背光拍摄
- ❌ 避免过度曝光或欠曝
- ❌ 避免运动模糊（提高快门速度）
- ✅ 确保标记点颜色鲜艳
- ✅ 多角度拍摄便于后续校准

---

## 常见问题

### Q1: 检测率很低（< 50%）

**可能原因**:
- 颜色 HSV 范围不准确
- 光照不足或不均匀
- 标记点太小或距离太远
- 标记点褪色或损坏

**解决方案**:
1. 使用 `src/tools/0_video_color_picker.py` 重新校准 HSV
2. 增加光照，重新拍摄
3. 打印更大的标记点（15cm × 15cm）
4. 更换新的标记点

---

### Q2: 背景干扰严重

**现象**: 过滤后仍有多条轨迹，或点数很少

**解决方案**:
1. 调整 HSV 范围，缩小颜色范围
2. 降低 `max_spatial_jump` 参数（如改为 50）
3. 提高 `min_segment_length` 参数（如改为 30）
4. 提高 `motion_threshold` 参数（如改为 10.0）
5. 更换背景色（与标记点对比度高的背景）

---

### Q3: 某些帧检测不到

**原因**: 
- 标记被遮挡
- 运动模糊
- 角度问题（标记点朝向背面）

**解决方案**:
- 正常现象，系统允许短暂中断（`max_frame_gap = 3`）
- 如中断过长，会自动分割为多段
- 增加标记点数量，确保至少 1 个可见
- 提高拍摄帧率，减少运动模糊

---

### Q4: 如何选择保留哪条轨迹？

**方法 1**: 查看统计信息
```bash
python src/core/2_detect_and_track.py --video "data/raw/video.mp4"
# 观察输出的运动幅度和覆盖率
```

**方法 2**: 查看 3D 图
```bash
python src/core/3_visualize_3d.py --video "data/raw/video.mp4"
# 平滑连续的曲线 = 真实标记点
```

**方法 3**: 手动指定颜色 ID
```bash
python src/core/4_export_csv.py --video "data/raw/video.mp4" --color-id 0
```

**方法 2**: 查看 3D 图
```bash
python src/core/3_visualize_3d.py --video "data/raw/video.mp4"
# 平滑连续的曲线 = 真实标记点
```

**方法 3**: 手动指定颜色 ID
```bash
python src/core/4_export_csv.py --video "data/raw/video.mp4" --color-id 0
```

---

### Q5: 如何处理多个标记点？

系统支持同时跟踪多个颜色的标记点：
1. 所有满足过滤条件的标记点都会保留
2. 导出 CSV 时包含所有标记点的数据
3. 后续可用于刚体配准（Kabsch 算法）

---

## 后续处理

### 1. 相机标定（可选）

如需获取真实 3D 坐标，需要标定相机：
- 使用 OpenCV 标定工具
- 获取内参矩阵和畸变系数
- 应用去畸变校正

### 2. 三角测量（多相机）

如有多个相机同步拍摄：
1. 每个相机分别导出 CSV
2. 使用三角测量重建 3D 坐标
3. 需要提前完成多相机标定

### 3. 刚体配准（Kabsch 算法）

从多个标记点重建刚体姿态：
1. 准备 Body 系模板点（标记点在刚体坐标系的位置）
2. 使用 Kabsch 算法求解旋转矩阵 R(t) 和平移向量 T(t)
3. 得到刚体的 6 自由度姿态

### 4. 角速度计算

从旋转矩阵序列计算角速度：
```python
# 伪代码
R_dot = (R[t+1] - R[t]) / dt
omega_skew = R_dot @ R[t].T
omega = [omega_skew[2,1], omega_skew[0,2], omega_skew[1,0]]
```

---

## 系统要求

### 软件环境
- **Python**: 3.7+
- **OpenCV**: 4.8.0+
- **NumPy**: 1.24.0+
- **Matplotlib**: 3.7.0+

### 硬件要求
- **RAM**: 8GB 推荐（4GB 可运行）
- **CPU**: 任意（单线程处理，暂不支持 GPU 加速）
- **硬盘**: 1GB+（存储输出文件）

### 操作系统
- Windows 7/10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+)

---

## 许可证

本项目用于贾尼别科夫效应科研实验。

---

## 技术支持

详细使用指南请参考 `docs/usage_guide.md`

**调试工具**:
- 颜色校准: `src/tools/0_color_calibration.py`
- 视频拾取: `src/tools/0_video_color_picker.py`
- 实时调试: `src/tools/0_realtime_debug.py`

**论文写作（光学课大作业模板）**:
- 论文模板（含占位与写作提示）: `docs/optics_term_paper_template.md`
- 参数表（独立填写）: `docs/parameter_tables.md`

---

## 版本历史

- **v1.0** - 初始版本，基础检测与跟踪功能
- **v1.1** - 新增运动幅度过滤
- **v1.2** - 新增批量处理功能
- **v1.3** - 优化降噪算法，提升检测率
