"""
批量处理视频文件夹中的所有视频
自动扫描videos目录，对每个视频执行检测跟踪
"""

import argparse
import subprocess
import sys
from pathlib import Path


def find_videos(video_dir):
    """查找所有视频文件"""
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    video_files = set()  # 使用set去重

    video_path = Path(video_dir)
    if not video_path.exists():
        print(f"Error: Directory does not exist: {video_dir}")
        return []

    for ext in video_extensions:
        video_files.update(video_path.glob(f"*{ext}"))
        video_files.update(video_path.glob(f"*{ext.upper()}"))

    return sorted(list(video_files))


def process_video(
    video_path,
    python_exe,
    script_path,
    export_csv=False,
    visualize_3d=False,
    render_video=False,
):
    """处理单个视频"""
    print("\n" + "=" * 80)
    print(f"Processing Video: {video_path.name}")
    print("=" * 80)

    # 1. 检测与跟踪
    print("\n[1/4] Detection and Tracking...")
    cmd = [python_exe, script_path, "--video", str(video_path)]
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"✗ Detection failed: {video_path.name}")
        return False

    # 2. 导出CSV（可选）
    if export_csv:
        print("\n[2/4] Exporting CSV data...")
        csv_script = Path(script_path).parent / "4_export_csv.py"
        cmd = [python_exe, str(csv_script), "--video", str(video_path)]
        subprocess.run(cmd, capture_output=False)

    # 3. 生成3D可视化（可选）
    if visualize_3d:
        print("\n[3/4] Generating 3D trajectory plot...")
        viz_script = Path(script_path).parent / "3_visualize_3d.py"
        cmd = [python_exe, str(viz_script), "--video", str(video_path)]
        subprocess.run(cmd, capture_output=False)

    # 4. 渲染检测视频（可选）
    if render_video:
        print("\n[4/4] Rendering detection video...")
        render_script = Path(script_path).parent / "5_render_detection_video.py"
        cmd = [
            python_exe,
            str(render_script),
            "--video",
            str(video_path),
            "--filter-motion",
        ]
        subprocess.run(cmd, capture_output=False)

    print(f"\n✓ Completed: {video_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch process all videos in the videos folder")
    parser.add_argument(
        "--video-dir", default="../videos", help="Video folder path (default: ../videos)"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python interpreter path (default: current)"
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="Export CSV data (default True)",
    )
    parser.add_argument(
        "--visualize-3d",
        action="store_true",
        default=True,
        help="Generate 3D strategy plot (default True)",
    )
    parser.add_argument(
        "--render-video",
        action="store_true",
        default=True,
        help="Render detection video (default True)",
    )
    parser.add_argument(
        "--skip-interactive",
        action="store_true",
        default=False,
        help="Skip interactive confirmation",
    )
    args = parser.parse_args()

    # 获取绝对路径
    script_dir = Path(__file__).parent
    video_dir = (script_dir / args.video_dir).resolve()
    detect_script = script_dir / "2_detect_and_track.py"

    if not detect_script.exists():
        print(f"Error: Detection script not found: {detect_script}")
        return

    # 查找所有视频
    print(f"Scanning video directory: {video_dir}")
    videos = find_videos(video_dir)

    if not videos:
        print("No video files found")
        print("Supported formats: .mp4, .avi, .mov, .mkv, .flv, .wmv")
        return

    print(f"\nFound {len(videos)} video files:")
    for i, video in enumerate(videos, 1):
        size_mb = video.stat().st_size / (1024 * 1024)
        print(f"  {i}. {video.name} ({size_mb:.1f} MB)")
    print(f"  {len(videos) + 1}. Process all videos")

    # 交互式选择
    if not args.skip_interactive:
        while True:
            try:
                choice = input(f"\nPlease select (1-{len(videos) + 1}): ").strip()
                idx = int(choice)

                if idx == len(videos) + 1:
                    # 处理全部
                    print("✓ Processing all videos")
                    break
                elif 1 <= idx <= len(videos):
                    # 处理单个视频
                    selected = videos[idx - 1]
                    print(f"✓ Selected: {selected.name}")
                    videos = [selected]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(videos) + 1}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nCancelled")
                return

    # 批量处理
    print("\nStarting batch processing...")
    success_count = 0
    failed_videos = []

    for i, video in enumerate(videos, 1):
        print(f"\nProgress: [{i}/{len(videos)}]")

        success = process_video(
            video,
            args.python,
            str(detect_script),
            args.export_csv,
            args.visualize_3d,
            args.render_video,
        )

        if success:
            success_count += 1
        else:
            failed_videos.append(video.name)

    # 汇总结果
    print("\n" + "=" * 80)
    print("Batch Processing Complete")
    print("=" * 80)
    print(f"Success: {success_count}/{len(videos)}")

    if failed_videos:
        print(f"\nFailed videos ({len(failed_videos)}):")
        for name in failed_videos:
            print(f"  - {name}")

    print("\nOutput locations:")
    print("  Trajectory CSV:  output/trajectories/")
    print("  3D Visualization: output/visualizations/")
    print("  Detection Video: output/videos/")
    print("\nNotes:")
    print("  - Run 1_generate_markers.py separately to generate marker dot images")
    print("  - Marker dot images will be saved in output/markers/")


if __name__ == "__main__":
    main()
