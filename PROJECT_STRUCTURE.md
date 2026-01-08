# 项目结构说明

本项目采用模块化的目录结构，便于代码维护和功能扩展。

## 目录结构

```
视觉方案/
├── src/                          # 源代码目录
│   ├── core/                     # 核心功能脚本
│   │   ├── 1_generate_markers.py      # 生成标记点图片
│   │   ├── 2_detect_and_track.py      # 检测与跟踪（主程序）
│   │   ├── 3_visualize_3d.py          # 3D 轨迹可视化
│   │   ├── 4_export_csv.py            # 导出 CSV 数据
│   │   ├── 5_render_detection_video.py # 渲染检测视频
│   │   └── utils.py                   # 工具函数库
│   └── tools/                    # 调试和校准工具
│       ├── 0_color_calibration.py   # 颜色校准工具
│       ├── 0_realtime_debug.py      # 实时调试工具
│       └── 0_video_color_picker.py  # 视频颜色拾取工具
├── scripts/                      # 安装和维护脚本
│   ├── batch_process_videos.py      # 批量处理视频
│   ├── 6_postprocess_trajectory.py  # 轨迹后处理
│   └── 7_prepare_paper_assets_12_16.py # 准备论文素材
├── config/                       # 配置文件
│   └── colors.json                   # 颜色定义和 HSV 参数
├── data/                         # 数据目录
│   ├── raw/                       # 原始视频（待处理）
│   └── processed/                 # 处理后的数据
│       ├── markers/                 # 生成的标记点图片
│       ├── trajectories/            # CSV 轨迹数据
│       ├── visualizations/          # 3D 可视化图片
│       └── videos/                  # 渲染的检测视频
├── docs/                         # 文档目录
│   ├── images/                     # 文档图片
│   ├── latex/                      # LaTeX 论文文件
│   ├── references/                 # 参考资料和笔记
│   └── optics_term_paper.md        # 论文文档（旧版本）
├── assets/                       # 演示文稿等资源
│   └── PPT.pdf                     # 演示文稿
├── reports/                      # 报告目录
│   └── optics_term_paper.md        # 视觉跟踪相关论文
├── requirements.txt               # Python 依赖
├── install.bat                    # Windows 安装脚本
├── README.md                      # 项目说明文档
└── .gitignore                     # Git 忽略配置
```

## 文件分类说明

### 源代码 (src/)
- **core/**: 核心功能脚本，用于主要的图像处理和分析任务
- **tools/**: 调试和校准工具，用于参数调整和实时测试

### 脚本 (scripts/)
- 批量处理脚本和维护工具

### 配置 (config/)
- 系统配置文件，包括颜色参数等

### 数据 (data/)
- **raw/**: 原始视频文件（待处理）
- **processed/**: 系统生成的各种输出结果

### 文档 (docs/)
- 项目文档、LaTeX 论文源文件和参考资料

### 资源 (assets/)
- 演示文稿、图片等资源文件

### 报告 (reports/)
- 研究报告和论文

## 使用说明

1. **准备视频**: 将待处理的视频文件放入 `data/raw/` 目录
2. **生成标记**: 运行 `python src/core/1_generate_markers.py` 生成标记点图片
3. **检测跟踪**: 运行 `python src/core/2_detect_and_track.py --video "data/raw/你的视频.mp4"` 进行检测
4. **查看结果**: 检测结果保存在 `data/processed/` 目录下

详细使用说明请参考 `README.md` 文档。
