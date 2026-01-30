# main.py
"""
축구 드리블 분석 시스템
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pose_extractor import PoseExtractor
from analysis.ball_motion_analyzer import BallMotionAnalyzer
from analysis.head_pose_analyzer import HeadPoseAnalyzer
from analysis.trunk_pose_analyzer import TrunkPoseAnalyzer
from visualization.skeleton_drawer import SkeletonDrawer
from visualization.ball_motion_plotter import BallMotionPlotter
from visualization.head_pose_plotter import HeadPosePlotter
from visualization.trunk_pose_plotter import TrunkPosePlotter
import cv2
import config


def main():
    print("\n" + "="*70)
    print("축구 드리블 분석 시스템")
    print("="*70)

    # 비디오 경로 설정
    video_path = "input/in_in/동시촬영_3.mp4"

    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"\n 오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        print(f"   input/ 폴더에 soccer1.mp4 파일을 넣어주세요.")
        return

    print(f"\n📹 비디오 파일: {video_path}")

    # 1. 비디오 정보 출력
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"   - 총 프레임: {total_frames}")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - 재생 시간: {duration:.1f}초")

    # 2. 포즈 추출
    print(f"\n{'='*70}")
    print("1단계: 포즈 추출 중...")
    print(f"{'='*70}")

    extractor = PoseExtractor(
        model_complexity=config.MEDIAPIPE_CONFIG['model_complexity'],
        min_detection_confidence=config.MEDIAPIPE_CONFIG['min_detection_confidence'],
        min_tracking_confidence=config.MEDIAPIPE_CONFIG['min_tracking_confidence'],
        # 공 탐지 설정
        ball_detector_config=config.BALL_DETECTION_CONFIG
    )

    pose_frames = extractor.extract_from_video(video_path)

    if len(pose_frames) == 0:
        print("\n 오류: 영상에서 포즈를 감지할 수 없습니다.")
        return

    print(f"\n포즈 추출 완료: {len(pose_frames)}개 프레임")

    # 3. 공 움직임 분석
    print(f"\n{'='*70}")
    print("2단계: 공 움직임 분석 중...")
    print(f"{'='*70}\n")

    ball_analyzer = BallMotionAnalyzer(
        min_velocity_threshold=config.BALL_MOTION_CONFIG['min_velocity_threshold'],
        peak_prominence=config.BALL_MOTION_CONFIG['peak_prominence'],
        min_distance_between_touches=config.BALL_MOTION_CONFIG['min_distance_between_touches'],
        use_smoothing=config.BALL_MOTION_CONFIG['use_smoothing'],
        smoothing_window=config.BALL_MOTION_CONFIG['smoothing_window']
    )
    ball_motion_data = ball_analyzer.analyze(pose_frames)

    if ball_motion_data:
        print(ball_motion_data)
    else:
        print("Warning: 공 움직임 분석 실패 (공이 충분히 탐지되지 않음)")

    # 3. 머리 자세 분석
    print(f"\n{'='*70}")
    print("3단계: 머리 자세 분석 중...")
    print(f"{'='*70}\n")

    head_analyzer = HeadPoseAnalyzer(min_visibility_threshold=0.5)
    head_pose_data = head_analyzer.analyze(pose_frames, ball_motion_data)

    if head_pose_data:
        print(head_pose_data)
    else:
        print("Warning: 머리 자세 분석 실패 (랜드마크 신뢰도 부족)")

    # 3-1. 상체 자세 분석
    print(f"\n{'='*70}")
    print("동시촬영_3 단계: 상체 자세 분석 중...")
    print(f"{'='*70}\n")

    trunk_analyzer = TrunkPoseAnalyzer(min_visibility_threshold=0.5)
    trunk_pose_data = trunk_analyzer.analyze(pose_frames)

    if trunk_pose_data:
        print(trunk_pose_data)
    else:
        print("Warning: 상체 자세 분석 실패 (랜드마크 신뢰도 부족)")

    # 4. 공 움직임 그래프 생성
    if ball_motion_data:
        print(f"\n{'='*70}")
        print("4단계: 공 움직임 그래프 생성 중...")
        print(f"{'='*70}\n")

        graphs_dir = Path('output/graphs')
        graphs_dir.mkdir(parents=True, exist_ok=True)

        plotter = BallMotionPlotter()
        plotter.plot_motion(ball_motion_data, save_path=str(graphs_dir / 'ball_motion 동시촬영_3.png'))

    # 4-1. 머리 각도 그래프 생성
    if head_pose_data:
        print(f"\n{'='*70}")
        print("4-1단계: 머리 각도 그래프 생성 중...")
        print(f"{'='*70}\n")

        graphs_dir = Path('output/graphs')
        graphs_dir.mkdir(parents=True, exist_ok=True)

        head_plotter = HeadPosePlotter()
        head_plotter.plot_head_angle(head_pose_data,
                                     save_path=str(graphs_dir / 'head_angle 동시촬영_3.png'))

    # 4-2. 상체 각도 그래프 생성
    if trunk_pose_data:
        print(f"\n{'='*70}")
        print("4-2단계: 상체 각도 그래프 생성 중...")
        print(f"{'='*70}\n")

        graphs_dir = Path('output/graphs')
        graphs_dir.mkdir(parents=True, exist_ok=True)

        trunk_plotter = TrunkPosePlotter()
        trunk_plotter.plot_trunk_angle(trunk_pose_data,
                                       save_path=str(graphs_dir / 'trunk_angle 동시촬영_3.png'))

    # 5. 스켈레톤 비디오 생성 (공 위치 및 터치 표시 포함)
    print(f"\n{'='*70}")
    print("5단계: 스켈레톤 비디오 생성 중...")
    print(f"{'='*70}\n")

    create_skeleton_video(video_path, pose_frames, ball_motion_data)

    print()


def create_skeleton_video(video_path: str, pose_frames, ball_motion_data=None):
    """
    스켈레톤 비디오 생성 (공 위치 및 터치 표시 포함)

    Args:
        video_path: 원본 비디오 경로
        pose_frames: 포즈 프레임 리스트
        ball_motion_data: BallMotionData 객체 (공 움직임 분석 결과)
    """
    videos_dir = Path('output/videos')
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 스켈레톤 드로워 초기화
    drawer = SkeletonDrawer(color=(0, 255, 0))  # 초록색

    # 비디오 정보
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 출력 비디오 설정
    output_path = str(videos_dir / 'skeleton_output 동시촬영_3.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("스켈레톤 그리는 중...")

    # pose_frames를 frame_number로 인덱싱 (빠른 검색용)
    pose_dict = {pf.frame_number: pf for pf in pose_frames}

    # 터치 프레임 세트 (빠른 검색용)
    touch_frames_set = set(ball_motion_data.touch_frames) if ball_motion_data else set()

    # 비디오 재시작
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # 현재 프레임 번호에 해당하는 포즈가 있으면 스켈레톤 그리기
        if frame_idx in pose_dict:
            pose_frame = pose_dict[frame_idx]
            landmarks = pose_frame.landmarks
            ball_position = pose_frame.ball_position
            ball_bbox = pose_frame.ball_bbox

            # 1. 스켈레톤 그리기
            frame = drawer.draw_skeleton(frame, landmarks)

            # 2. 공 바운딩 박스 그리기
            if ball_bbox is not None:
                frame = drawer.draw_ball_bbox(frame, ball_bbox)

            # 3. 터치 순간 하이라이트
            if frame_idx in touch_frames_set:
                frame = drawer.draw_touch_highlight(frame, ball_position)

        out.write(frame)
        frame_idx += 1

    # 정리
    cap.release()
    out.release()

    print(f" 스켈레톤 비디오 저장: {output_path}")
    print(f"   총 {frame_idx}개 프레임 중 {len(pose_frames)}개 프레임에서 포즈 감지")


if __name__ == "__main__":
    main()
