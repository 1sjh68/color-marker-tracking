"""
Prepare paper assets for the latest multi-camera experiment (12.16).

What it does:
1) Copy key tables/figures from `output/trajectories/12.16` into `docs/images/`
2) Generate a montage image from `reconstruction_dynamic.mp4` via ffmpeg (if available)
3) Print a concise summary parsed from `comparison_report.txt` and `diagnostic_report.txt`

This script only uses the standard library. It does not (re)run triangulation/optimization.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DiagnosticSummary:
    pose_frames: int | None = None
    raw_points: int | None = None
    complete_points_hist: dict[int, int] | None = None  # {num_points: frames}


@dataclass(frozen=True)
class ComparisonSummary:
    poses_path: str | None = None
    total_frames: int | None = None
    duration_s: float | None = None
    mass_kg: float | None = None
    dims_m: tuple[float, float, float] | None = None
    inertia: tuple[float, float, float] | None = None
    omega0: tuple[float, float, float] | None = None
    damping: tuple[float, float, float] | None = None
    mae_rad_s: float | None = None
    exp_energy_change_pct: float | None = None
    sim_energy_change_pct: float | None = None


def parse_diagnostic_report(path: Path) -> DiagnosticSummary:
    if not path.exists():
        return DiagnosticSummary()

    text = path.read_text(encoding="utf-8", errors="replace")
    pose_frames = None
    raw_points = None
    hist: dict[int, int] = {}

    m = re.search(r"姿态数据帧数:\s*(\d+)", text)
    if m:
        pose_frames = int(m.group(1))

    m = re.search(r"原始轨迹点数:\s*(\d+)", text)
    if m:
        raw_points = int(m.group(1))

    for m in re.finditer(r"(\d+)点:\s*(\d+)\s*帧", text):
        hist[int(m.group(1))] = int(m.group(2))

    return DiagnosticSummary(
        pose_frames=pose_frames,
        raw_points=raw_points,
        complete_points_hist=hist or None,
    )


def parse_comparison_report(path: Path) -> ComparisonSummary:
    if not path.exists():
        return ComparisonSummary()

    text = path.read_text(encoding="utf-8", errors="replace")

    def _float_triplet(pattern: str) -> tuple[float, float, float] | None:
        m = re.search(pattern, text)
        if not m:
            return None
        return (float(m.group(1)), float(m.group(2)), float(m.group(3)))

    poses_path = None
    m = re.search(r"数据源:\s*(.+)\s*$", text, flags=re.MULTILINE)
    if m:
        poses_path = m.group(1).strip()

    total_frames = None
    duration_s = None
    m = re.search(r"总帧数:\s*(\d+)\s*\(时长:\s*([0-9.]+)s\)", text)
    if m:
        total_frames = int(m.group(1))
        duration_s = float(m.group(2))

    mass_kg = None
    m = re.search(r"质量:\s*([0-9.]+)\s*kg", text)
    if m:
        mass_kg = float(m.group(1))

    dims_m = _float_triplet(r"尺寸:\s*\(([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\)\s*m")
    inertia = _float_triplet(r"惯量:\s*I1=([0-9.eE+-]+),\s*I2=([0-9.eE+-]+),\s*I3=([0-9.eE+-]+)")
    omega0 = _float_triplet(
        r"最佳初始角速度:\s*\[\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\]"
    )
    damping = _float_triplet(
        r"拟合阻尼系数:\s*\[\s*([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s*\]"
    )

    mae_rad_s = None
    m = re.search(r"角速度平均绝对误差\s*\(MAE\):\s*([0-9.]+)\s*rad/s", text)
    if m:
        mae_rad_s = float(m.group(1))

    exp_energy_change_pct = None
    m = re.search(r"实验能量损耗:\s*([0-9.+-]+)%", text)
    if m:
        exp_energy_change_pct = float(m.group(1))

    sim_energy_change_pct = None
    m = re.search(r"模拟能量损耗:\s*([0-9.+-]+)%", text)
    if m:
        sim_energy_change_pct = float(m.group(1))

    return ComparisonSummary(
        poses_path=poses_path,
        total_frames=total_frames,
        duration_s=duration_s,
        mass_kg=mass_kg,
        dims_m=dims_m,
        inertia=inertia,
        omega0=omega0,
        damping=damping,
        mae_rad_s=mae_rad_s,
        exp_energy_change_pct=exp_energy_change_pct,
        sim_energy_change_pct=sim_energy_change_pct,
    )


def try_generate_montage(ffmpeg: str, video_path: Path, out_path: Path) -> bool:
    if not video_path.exists():
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: avoid quotes; some shells treat quotes literally in this environment.
    vf = r"select=expr=not(mod(n\,30)),scale=240:-1,tile=8x2"
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        "1",
        str(out_path),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return True
    except FileNotFoundError:
        return False
    except subprocess.CalledProcessError:
        return False


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 12.16 experiment assets for the paper")
    parser.add_argument(
        "--src-dir",
        default=str(Path("output") / "trajectories" / "12.16"),
        help="Source directory (default: output/trajectories/12.16)",
    )
    parser.add_argument(
        "--dst-dir",
        default=str(Path("docs") / "images"),
        help="Destination directory (default: docs/images)",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="ffmpeg executable name/path (default: ffmpeg)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)

    # Copy figures used by the paper.
    copies: list[tuple[Path, Path]] = [
        (src_dir / "trajectory_3d_reconstruction.png", dst_dir / "trajectory_3d_reconstruction.png"),
        (src_dir / "reconstructed_3d_plot.png", dst_dir / "reconstructed_3d_plot.png"),
        (src_dir / "omega_time_series.png", dst_dir / "omega_time_series.png"),
        (src_dir / "gravity_analysis.png", dst_dir / "gravity_analysis.png"),
        (src_dir / "energy_conservation.png", dst_dir / "energy_conservation.png"),
        (src_dir / "angular_momentum_conservation.png", dst_dir / "angular_momentum_conservation.png"),
        (src_dir / "phase_space_comparison.png", dst_dir / "phase_space_comparison.png"),
        (src_dir / "polhode_trajectory.png", dst_dir / "polhode_trajectory.png"),
        (src_dir / "theory_vs_experiment.png", dst_dir / "theory_vs_experiment.png"),
        (src_dir / "calibration_candidates.png", dst_dir / "calibration_candidates.png"),
        (src_dir / "z_comparison.png", dst_dir / "z_comparison.png"),
        (src_dir / "euler_phase_space.png", dst_dir / "euler_phase_space.png"),
        (src_dir / "flip_angle_analysis.png", dst_dir / "flip_angle_analysis.png"),
        (src_dir / "energy_comparison.png", dst_dir / "energy_comparison.png"),
        (src_dir / "comprehensive_comparison.png", dst_dir / "comprehensive_comparison.png"),
    ]

    theory_dir = src_dir / "theory_comparison"
    copies.extend(
        [
            (theory_dir / "3d_phase_space.png", dst_dir / "3d_phase_space.png"),
            (theory_dir / "quaternion_evolution.png", dst_dir / "quaternion_evolution.png"),
            (theory_dir / "period_analysis.png", dst_dir / "period_analysis.png"),
            (theory_dir / "inertia_sensitivity.png", dst_dir / "inertia_sensitivity.png"),
        ]
    )

    copied = 0
    for src, dst in copies:
        if copy_if_exists(src, dst):
            copied += 1

    # Generate montage (optional).
    montage_ok = try_generate_montage(
        ffmpeg=str(args.ffmpeg),
        video_path=src_dir / "reconstruction_dynamic.mp4",
        out_path=dst_dir / "reconstruction_montage.png",
    )

    # Parse reports (optional).
    diag = parse_diagnostic_report(src_dir / "diagnostic_report.txt")
    cmp = parse_comparison_report(src_dir / "comparison_report.txt")

    print("=== 12.16 Paper Assets ===")
    print(f"src_dir: {src_dir}")
    print(f"dst_dir: {dst_dir}")
    print(f"copied_figures: {copied}")
    print(f"montage: {'ok' if montage_ok else 'skipped'}")
    print("")
    print("=== Diagnostic Summary ===")
    if diag.pose_frames is not None:
        print(f"pose_frames: {diag.pose_frames}")
    if diag.raw_points is not None:
        print(f"raw_points: {diag.raw_points}")
    if diag.complete_points_hist:
        total = sum(diag.complete_points_hist.values())
        for k in sorted(diag.complete_points_hist.keys()):
            v = diag.complete_points_hist[k]
            pct = 100.0 * float(v) / float(total) if total else 0.0
            print(f"- complete_3d_points={k}: frames={v} ({pct:.1f}%)")
    print("")
    print("=== Comparison Summary ===")
    if cmp.poses_path:
        print(f"poses_path: {cmp.poses_path}")
    if cmp.total_frames is not None and cmp.duration_s is not None and cmp.duration_s > 0:
        fps_est = float(cmp.total_frames - 1) / float(cmp.duration_s)
        print(f"total_frames: {cmp.total_frames}, duration_s: {cmp.duration_s:.3f}, fps_est≈{fps_est:.1f}")
    if cmp.mass_kg is not None:
        print(f"mass_kg: {cmp.mass_kg}")
    if cmp.dims_m is not None:
        print(f"dims_m: {cmp.dims_m}")
    if cmp.inertia is not None:
        print(f"inertia: {cmp.inertia}")
    if cmp.omega0 is not None:
        print(f"omega0: {cmp.omega0}")
    if cmp.damping is not None:
        print(f"damping: {cmp.damping}")
    if cmp.mae_rad_s is not None:
        print(f"mae_rad_s: {cmp.mae_rad_s}")
    if cmp.exp_energy_change_pct is not None:
        print(f"exp_energy_change_pct: {cmp.exp_energy_change_pct}%")
    if cmp.sim_energy_change_pct is not None:
        print(f"sim_energy_change_pct: {cmp.sim_energy_change_pct}%")


if __name__ == "__main__":
    main()

