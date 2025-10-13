# main.py
"""
=============================================================================
Soccer Motion Analysis - Main Script
=============================================================================

두 축구 드리블 영상의 모션 비교 파이프라인

전체 실행 흐름:
1. extract_motion_data(): 비디오 → 각도 데이터
2. compare_motions(): 두 데이터 통계 비교
3. save_results(): CSV + 텍스트 리포트 저장
4. create_visualizations(): 그래프 생성
5. create_skeleton_videos(): 스켈레톤 비디오 생성

실행 방법:
    $ python main.py

출력:
    output/
    ├── data/comparison_table.csv
    ├── reports/comparison_report.txt
    ├── plots/*.png
    └── videos/skeleton_*.mp4

설정 변경:
    - config.py에서 MediaPipe 설정, 색상 등 수정 가능
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pose_extractor import PoseExtractor
from core.video_processor import VideoProcessor
from analysis.segment_analyzer import SegmentAnalyzer
from analysis.joint_analyzer import JointAnalyzer
from analysis.comparison import MotionComparison
from visualization.angle_plotter import AnglePlotter
from visualization.skeleton_drawer import SkeletonDrawer
import cv2
import config


def extract_motion_data(video_path: str, video_name: str):
    """
    비디오에서 모션 데이터 추출 - 파이프라인의 첫 번째 단계

    처리 과정:
    1. VideoProcessor로 비디오 정보 확인
    2. PoseExtractor로 모든 프레임에서 포즈 추출
    3. SegmentAnalyzer로 분절 각도 계산 (7개)
    4. JointAnalyzer로 관절 각도 계산 (6개)
    5. 각 프레임의 각도들을 배열로 수집

    Args:
        video_path: 비디오 파일 경로 (예: "input/soccer1.mp4")
        video_name: 비디오 이름 (표시용, 예: "Soccer 1")

    Returns:
        dict: 모든 관절/분절 각도 데이터
            {
                'left_knee_angles': np.array([152, 148, ...]),  # 프레임 수만큼
                'trunk_angles': np.array([85, 84, ...]),
                ...  # 총 13개 항목 (7 segments + 6 joints)
            }
            None: 포즈 추출 실패 시

    데이터 흐름:
        비디오 파일
            ↓ [PoseExtractor]
        List[PoseFrame] (프레임별 landmark)
            ↓ [SegmentAnalyzer + JointAnalyzer]
        각 프레임마다 13개 각도 계산
            ↓ [수집 및 배열화]
        motion_data (딕셔너리)
    """
    print(f"\n{'='*70}")
    print(f"Processing: {video_name}")
    print(f"{'='*70}")
    
    # 1. 비디오 정보 확인
    video_proc = VideoProcessor(video_path)
    video_info = video_proc.get_video_info()
    print(f"Video Info: {video_info['total_frames']} frames, {video_info['fps']:.1f} FPS, {video_info['duration']:.1f}s")
    
    # 2. 포즈 추출
    print("\n[1/3] Extracting poses from video...")
    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence']
    )
    
    pose_frames = extractor.extract_from_video(video_path)
    print(f"✓ Extracted {len(pose_frames)} frames with valid poses")
    
    if len(pose_frames) == 0:
        print("ERROR: No poses detected in video!")
        return None
    
    # 3. 각도 계산
    print("\n[2/3] Calculating angles...")
    segment_analyzer = SegmentAnalyzer()
    joint_analyzer = JointAnalyzer()
    
    motion_data = {
        # 분절 각도
        'trunk_angles': [],
        'left_thigh_angles': [],
        'right_thigh_angles': [],
        'left_shank_angles': [],
        'right_shank_angles': [],
        'left_foot_angles': [],
        'right_foot_angles': [],
        
        # 관절 각도
        'left_knee_angles': [],
        'right_knee_angles': [],
        'left_hip_angles': [],
        'right_hip_angles': [],
        'left_ankle_angles': [],
        'right_ankle_angles': []
    }
    
    for frame in pose_frames:
        # World landmarks 사용 (실제 3D 좌표)
        landmarks = frame.world_landmarks
        
        # 분절 각도 계산
        segments = segment_analyzer.calculate_all_segments(landmarks)
        for seg_name, angle in segments.items():
            motion_data[f'{seg_name}_angles'].append(angle)
        
        # 관절 각도 계산
        joints = joint_analyzer.calculate_all_joints(landmarks)
        for joint_name, angle in joints.items():
            motion_data[f'{joint_name}_angles'].append(angle)
    
    # Numpy 배열로 변환
    for key in motion_data:
        motion_data[key] = np.array(motion_data[key])
    
    print("✓ Angle calculation complete")
    
    # 4. 기본 통계 출력
    print("\n[3/3] Basic Statistics:")
    print(f"Left Knee  : Mean={np.mean(motion_data['left_knee_angles']):.1f}°, "
          f"Range={np.max(motion_data['left_knee_angles']) - np.min(motion_data['left_knee_angles']):.1f}°")
    print(f"Right Knee : Mean={np.mean(motion_data['right_knee_angles']):.1f}°, "
          f"Range={np.max(motion_data['right_knee_angles']) - np.min(motion_data['right_knee_angles']):.1f}°")
    print(f"Trunk      : Mean={np.mean(motion_data['trunk_angles']):.1f}°, "
          f"Range={np.max(motion_data['trunk_angles']) - np.min(motion_data['trunk_angles']):.1f}°")
    
    return motion_data


def compare_motions(motion1_data, motion2_data, video1_name="Video 1", video2_name="Video 2"):
    """
    두 모션 비교
    
    Args:
        motion1_data: 첫 번째 모션 데이터
        motion2_data: 두 번째 모션 데이터
        video1_name: 첫 번째 비디오 이름
        video2_name: 두 번째 비디오 이름
        
    Returns:
        tuple: (comparison_table, summary_text)
    """
    print(f"\n{'='*70}")
    print(f"Comparing Motions: {video1_name} vs {video2_name}")
    print(f"{'='*70}\n")
    
    comparator = MotionComparison(motion1_data, motion2_data, video1_name, video2_name)
    
    # 1. 비교 테이블
    print("[1/2] Generating comparison table...")
    comparison_table = comparator.compare_characteristics()
    
    # 2. 스타일 요약
    print("[2/2] Generating style summary...")
    summary_text = comparator.generate_style_summary()
    
    return comparison_table, summary_text


def save_results(comparison_table, summary_text, output_dir='output'):
    """
    결과 저장
    
    Args:
        comparison_table: 비교 테이블
        summary_text: 요약 텍스트
        output_dir: 출력 디렉토리
    """
    print(f"\n{'='*70}")
    print("Saving Results")
    print(f"{'='*70}\n")
    
    # 디렉토리 생성
    data_dir = Path(output_dir) / 'data'
    reports_dir = Path(output_dir) / 'reports'
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. CSV 저장
    csv_path = data_dir / 'comparison_table.csv'
    comparison_table.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved comparison table to {csv_path}")
    
    # 2. 텍스트 리포트 저장
    report_path = reports_dir / 'comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"✓ Saved text report to {report_path}")


def create_visualizations(motion1_data, motion2_data, 
                         video1_name="Video 1", video2_name="Video 2",
                         output_dir='output'):
    """
    시각화 생성
    
    Args:
        motion1_data: 첫 번째 모션 데이터
        motion2_data: 두 번째 모션 데이터
        video1_name: 첫 번째 비디오 이름
        video2_name: 두 번째 비디오 이름
        output_dir: 출력 디렉토리
    """
    print(f"\n{'='*70}")
    print("Creating Visualizations")
    print(f"{'='*70}\n")
    
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plotter = AnglePlotter()
    
    # 주요 관절들 그래프 생성
    joints_to_plot = ['left_knee', 'right_knee', 'trunk']
    
    for joint in joints_to_plot:
        angle_key = f'{joint}_angles'
        if angle_key in motion1_data and angle_key in motion2_data:
            print(f"Creating plot for {joint}...")
            
            # 1. 시계열 비교 그래프
            plotter.plot_comparison(
                motion1_data[angle_key],
                motion2_data[angle_key],
                title=f'{joint.replace("_", " ").title()} Angle Comparison',
                labels=[video1_name, video2_name],
                save_path=str(plots_dir / f'{joint}_comparison.png')
            )
            
            # 2. 분포 히스토그램
            plotter.plot_distribution(
                motion1_data[angle_key],
                motion2_data[angle_key],
                title=f'{joint.replace("_", " ").title()} Angle Distribution',
                labels=[video1_name, video2_name],
                save_path=str(plots_dir / f'{joint}_distribution.png')
            )
    
    print(f"✓ All plots saved to {plots_dir}")


def create_skeleton_video_single(video_path, video_name, color, output_dir='output'):
    """
    단일 비디오에 스켈레톤을 그린 비디오 생성

    Args:
        video_path: 비디오 경로
        video_name: 비디오 이름
        color: 스켈레톤 색상
        output_dir: 출력 디렉토리

    Returns:
        str: 생성된 비디오 경로
    """
    videos_dir = Path(output_dir) / 'videos'
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 포즈 추출기 초기화
    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence']
    )

    # 스켈레톤 드로워 초기화
    drawer = SkeletonDrawer(color=color)

    print(f"Processing {video_name}...")
    print("  - Extracting poses...")

    # 포즈 추출
    pose_frames = extractor.extract_from_video(video_path)

    # 비디오 정보
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 비디오 설정 (원본 해상도 유지)
    output_filename = f'skeleton_{video_name.lower().replace(" ", "_")}.mp4'
    output_path = str(videos_dir / output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("  - Drawing skeleton...")

    # 비디오 재시작
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while frame_idx < len(pose_frames):
        ret, frame = cap.read()

        if not ret:
            break

        # 스켈레톤 그리기
        if frame_idx < len(pose_frames):
            landmarks = pose_frames[frame_idx].landmarks
            frame = drawer.draw_skeleton(frame, landmarks)

            # 비디오 이름 추가
            frame = drawer.draw_angle_text(frame, video_name, (10, 30))

        out.write(frame)
        frame_idx += 1

    # 정리
    cap.release()
    out.release()

    print(f"  ✓ Saved to {output_path}")
    return output_path


def create_skeleton_videos(video1_path, video2_path,
                           video1_name="Video 1", video2_name="Video 2",
                           output_dir='output'):
    """
    각 비디오에 대해 개별적으로 스켈레톤 비디오 생성

    Args:
        video1_path: 첫 번째 비디오 경로
        video2_path: 두 번째 비디오 경로
        video1_name: 첫 번째 비디오 이름
        video2_name: 두 번째 비디오 이름
        output_dir: 출력 디렉토리
    """
    print(f"\n{'='*70}")
    print("Creating Skeleton Videos")
    print(f"{'='*70}\n")

    # 각 비디오 개별 처리 (원본 해상도 유지)
    create_skeleton_video_single(
        video1_path,
        video1_name,
        config.VISUALIZATION_CONFIG['video1_color'],
        output_dir
    )

    print()

    create_skeleton_video_single(
        video2_path,
        video2_name,
        config.VISUALIZATION_CONFIG['video2_color'],
        output_dir
    )

    print(f"\n✓ All skeleton videos created in {output_dir}/videos/")


def main():
    """
    메인 실행 함수 - 전체 파이프라인 실행

    실행 순서:
    1. 비디오 파일 존재 확인
    2. extract_motion_data() × 2 (각 비디오)
       → motion1_data, motion2_data
    3. compare_motions()
       → comparison_table, summary_text
    4. 결과 출력 (콘솔)
    5. save_results()
       → CSV + 텍스트 파일
    6. create_visualizations()
       → PNG 그래프들
    7. create_skeleton_videos()
       → MP4 비디오들

    실행:
        $ python main.py

    입력 파일:
        - input/soccer1.mp4
        - input/soccer2.mp4

    출력 파일:
        - output/data/comparison_table.csv
        - output/reports/comparison_report.txt
        - output/plots/left_knee_comparison.png (등 6개)
        - output/videos/skeleton_soccer_1.mp4
        - output/videos/skeleton_soccer_2.mp4
    """
    print("\n" + "="*70)
    print("Soccer Motion Analysis - Dribbling Comparison")
    print("="*70)
    
    # 비디오 경로 설정
    video1_path = "input/soccer1.mp4"
    video2_path = "input/soccer2.mp4"
    
    # 파일 존재 확인
    if not os.path.exists(video1_path):
        print(f"ERROR: Video 1 not found: {video1_path}")
        return
    if not os.path.exists(video2_path):
        print(f"ERROR: Video 2 not found: {video2_path}")
        return
    
    # 1. 모션 데이터 추출
    motion1_data = extract_motion_data(video1_path, "Soccer 1")
    if motion1_data is None:
        return
    
    motion2_data = extract_motion_data(video2_path, "Soccer 2")
    if motion2_data is None:
        return
    
    # 2. 모션 비교
    comparison_table, summary_text = compare_motions(
        motion1_data, motion2_data,
        "Soccer 1", "Soccer 2"
    )
    
    # 3. 결과 출력
    print("\n" + "="*70)
    print("COMPARISON TABLE")
    print("="*70)
    print(comparison_table.to_string(index=False))
    
    print("\n" + summary_text)
    
    # 4. 결과 저장
    save_results(comparison_table, summary_text)
    
    # 5. 시각화 생성
    create_visualizations(motion1_data, motion2_data, "Soccer 1", "Soccer 2")

    # 6. 스켈레톤 비디오 생성
    create_skeleton_videos(video1_path, video2_path, "Soccer 1", "Soccer 2")

    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - output/data/comparison_table.csv")
    print(f"  - output/reports/comparison_report.txt")
    print(f"  - output/plots/*.png")
    print(f"  - output/videos/skeleton_soccer_1.mp4")
    print(f"  - output/videos/skeleton_soccer_2.mp4")
    print()


if __name__ == "__main__":
    main()