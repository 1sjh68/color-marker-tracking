"""
轨迹后处理与拟合（用于论文图表生成）

输入：scripts/4_export_csv.py 导出的 CSV（frame_idx,color_id,u,v）
输出：若干张分析图（默认写入 docs/images/）
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def setup_chinese_font() -> None:
    """配置 matplotlib 中文字体（尽量避免中文乱码/警告）。"""
    import platform

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
    plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class TrajectorySeries:
    frames: np.ndarray  # shape: (N,)
    u: np.ndarray  # shape: (N,)
    v: np.ndarray  # shape: (N,)

    @property
    def t(self) -> np.ndarray:
        raise RuntimeError("Use t = (frames - 1) / fps")


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def load_trajectories(csv_path: str) -> dict[int, TrajectorySeries]:
    points: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame_idx", "color_id", "u", "v"}
        if set(reader.fieldnames or []) != required:
            raise ValueError(
                f"CSV header must be exactly {sorted(required)}, got: {reader.fieldnames}"
            )

        for row in reader:
            frame = int(row["frame_idx"])
            color_id = int(row["color_id"])
            u = float(row["u"])
            v = float(row["v"])
            points[color_id].append((frame, u, v))

    trajectories: dict[int, TrajectorySeries] = {}
    for color_id, items in points.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        frames = np.array([x[0] for x in items_sorted], dtype=int)
        u = np.array([x[1] for x in items_sorted], dtype=float)
        v = np.array([x[2] for x in items_sorted], dtype=float)
        trajectories[color_id] = TrajectorySeries(frames=frames, u=u, v=v)
    return trajectories


def compute_center_trajectory(
    trajectories: dict[int, TrajectorySeries],
) -> TrajectorySeries:
    frame_to_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for series in trajectories.values():
        for frame, u, v in zip(series.frames, series.u, series.v, strict=True):
            frame_to_points[int(frame)].append((float(u), float(v)))

    frames = np.array(sorted(frame_to_points.keys()), dtype=int)
    u_center = np.array(
        [np.mean([p[0] for p in frame_to_points[int(f)]]) for f in frames], dtype=float
    )
    v_center = np.array(
        [np.mean([p[1] for p in frame_to_points[int(f)]]) for f in frames], dtype=float
    )
    return TrajectorySeries(frames=frames, u=u_center, v=v_center)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_uv_time_plot(
    out_path: str,
    trajectories: dict[int, TrajectorySeries],
    center: TrajectorySeries,
    fps: float,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for color_id, series in sorted(trajectories.items(), key=lambda x: x[0]):
        t = (series.frames - 1) / fps
        axes[0].plot(t, series.u, linewidth=1.2, label=f"ID {color_id}")
        axes[1].plot(t, series.v, linewidth=1.2, label=f"ID {color_id}")

    t_center = (center.frames - 1) / fps
    axes[0].plot(t_center, center.u, "k--", linewidth=1.6, label="Center (mean)")
    axes[1].plot(t_center, center.v, "k--", linewidth=1.6, label="Center (mean)")

    axes[0].set_ylabel("u (px)")
    axes[1].set_ylabel("v (px)")
    axes[1].set_xlabel("time (s)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(loc="best", ncol=3, fontsize=10)

    fig.suptitle("轨迹时间序列（像素域）", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_uv_path_plot(
    out_path: str,
    trajectories: dict[int, TrajectorySeries],
    center: TrajectorySeries,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))

    for color_id, series in sorted(trajectories.items(), key=lambda x: x[0]):
        ax.plot(series.u, series.v, linewidth=1.2, label=f"ID {color_id}")

    ax.plot(center.u, center.v, "k--", linewidth=1.6, label="Center (mean)")

    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title("轨迹投影（u-v 平面）", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    ax.invert_yaxis()  # 图像坐标系：向下为正

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gravity_fit_plot(
    out_path: str,
    center: TrajectorySeries,
    fps: float,
    frame_start: int | None,
    frame_end: int | None,
    smooth_window: int,
) -> dict[str, float]:
    t = (center.frames - 1) / fps
    v_raw = center.v
    v_smooth = moving_average(v_raw, smooth_window)

    mask = np.ones_like(center.frames, dtype=bool)
    if frame_start is not None:
        mask &= center.frames >= int(frame_start)
    if frame_end is not None:
        mask &= center.frames <= int(frame_end)

    t_fit = t[mask]
    v_fit = v_smooth[mask]
    if len(t_fit) < 10:
        raise ValueError("Not enough points for fitting; adjust --frame-start/--frame-end")

    coeff = np.polyfit(t_fit, v_fit, deg=2)  # v = a t^2 + b t + c
    a, b, c = (float(coeff[0]), float(coeff[1]), float(coeff[2]))
    v_model = np.polyval(coeff, t_fit)
    residual = v_fit - v_model

    g_px = 2.0 * a  # px/s^2（图像坐标系向下为正）
    rmse = float(np.sqrt(np.mean(residual**2)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(t, v_raw, color="#999999", linewidth=1.0, label="v_center (raw)")
    axes[0].plot(t, v_smooth, color="#1f77b4", linewidth=1.6, label=f"v_center (MA, w={smooth_window})")
    axes[0].plot(t_fit, v_model, "r-", linewidth=2.0, label=f"Quadratic fit: g≈{g_px:.2f} px/s²")
    axes[0].set_ylabel("v (px)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=10)

    axes[1].plot(t_fit, residual, "k-", linewidth=1.2)
    axes[1].axhline(0.0, color="r", linewidth=1.0, alpha=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("residual (px)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("抛体运动拟合（像素域：v 随时间的二次拟合）", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"a": a, "b": b, "c": c, "g_px_s2": float(g_px), "rmse_px": rmse}


def main() -> None:
    setup_chinese_font()

    parser = argparse.ArgumentParser(description="Post-process trajectories and generate plots")
    parser.add_argument("--csv", required=True, help="Trajectory CSV path (frame_idx,color_id,u,v)")
    parser.add_argument("--fps", type=float, default=60.0, help="Video FPS (default: 60)")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "docs", "images"),
        help="Output directory for figures (default: docs/images)",
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=80,
        help="Fit start frame index (default: 80)",
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        default=420,
        help="Fit end frame index (default: 420)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Moving-average window for fitting (default: 11)",
    )
    args = parser.parse_args()

    trajectories = load_trajectories(args.csv)
    if not trajectories:
        raise ValueError("No trajectories found in CSV")

    center = compute_center_trajectory(trajectories)

    out_dir = args.out_dir
    ensure_dir(out_dir)

    uv_time_path = os.path.join(out_dir, "trajectory_uv_vs_time.png")
    uv_path_path = os.path.join(out_dir, "trajectory_uv_path.png")
    gravity_fit_path = os.path.join(out_dir, "trajectory_gravity_fit.png")

    save_uv_time_plot(uv_time_path, trajectories, center, fps=float(args.fps))
    save_uv_path_plot(uv_path_path, trajectories, center)
    fit_stats = save_gravity_fit_plot(
        gravity_fit_path,
        center,
        fps=float(args.fps),
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        smooth_window=int(args.smooth_window),
    )

    # 输出简单统计，便于写入论文
    max_frame = int(max(series.frames.max() for series in trajectories.values()))
    print("=== Trajectory Summary ===")
    print(f"csv: {args.csv}")
    print(f"fps: {float(args.fps):.2f}")
    print(f"max_frame_in_csv: {max_frame}")
    for color_id, series in sorted(trajectories.items(), key=lambda x: x[0]):
        unique_frames = int(len(np.unique(series.frames)))
        coverage = 100.0 * unique_frames / float(max_frame)
        print(f"- ID {color_id}: points={len(series.frames)}, unique_frames={unique_frames}, coverage≈{coverage:.1f}%")
    print("")
    print("=== Gravity Fit (pixel domain) ===")
    print(f"fit_frames: [{args.frame_start}, {args.frame_end}]")
    print(f"g≈{fit_stats['g_px_s2']:.2f} px/s^2, RMSE≈{fit_stats['rmse_px']:.2f} px")
    print("")
    print("=== Figures ===")
    print(uv_time_path)
    print(uv_path_path)
    print(gravity_fit_path)


if __name__ == "__main__":
    main()

