# main.py
"""
Soccer Motion Analysis - Main Script
두 축구 드리블 영상의 모션 비교
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
import config


def extract_motion_data(video_path: str, video_name: str):
    """
    비디오에서 모션 데이터 추출
    
    Args:
        video_path: 비디오 파일 경로
        video_name: 비디오 이름
        
    Returns:
        dict: 모든 관절/분절 각도 데이터
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


def main():
    """
    메인 실행 함수
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
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"\nResults saved to:")
    print(f"  - output/data/comparison_table.csv")
    print(f"  - output/reports/comparison_report.txt")
    print(f"  - output/plots/*.png")
    print()


if __name__ == "__main__":
    main()