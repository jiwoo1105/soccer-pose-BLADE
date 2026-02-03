# main.py
"""
축구 드리블 분석 시스템 v2.0

BLADE 기반 3D 포즈 추출 + 4가지 평가 기준 통합 점수

파이프라인:
1. BLADE로 SMPL-X 54개 관절 3D 포즈 추출
2. 하이브리드 공 탐지 (YOLOv9 + 스켈레톤 폴백)
3. 터치 감지 (속도 + 거리 교차 검증)
4. 4가지 기준 평가 (헤드업, 상체 기울기, 무릎 업다운, 정렬)
5. 종합 점수 및 개선 제안
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core modules
from core.pose_extractor import PoseExtractor
from core.touch_detector import TouchDetector

# Analysis modules
from analysis.ball_motion_analyzer import BallMotionAnalyzer

# Scoring modules
from scoring import DribbleScorer

# Visualization modules
from visualization.skeleton_drawer import SkeletonDrawer
from visualization.ball_motion_plotter import BallMotionPlotter

import cv2
import numpy as np
import config


def main():
    print("\n" + "="*70)
    print("축구 드리블 분석 시스템 v2.0")
    print("BLADE 3D 포즈 + 4가지 평가 기준")
    print("="*70)

    # 비디오 경로 설정
    video_path = "input/in_in/동시촬영_3.mp4"

    # 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"\n오류: 비디오 파일을 찾을 수 없습니다: {video_path}")
        print(f"   input/ 폴더에 비디오 파일을 넣어주세요.")
        return

    print(f"\n비디오 파일: {video_path}")

    # 1. 비디오 정보 출력
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    print(f"   - 총 프레임: {total_frames}")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - 재생 시간: {duration:.1f}초")

    # 2. BLADE 3D 포즈 추출
    print(f"\n{'='*70}")
    print("1단계: BLADE 3D 포즈 추출 중...")
    print(f"{'='*70}")

    extractor = PoseExtractor(
        detect_ball=config.BLADE_CONFIG.get('enable_ball_detection', True),
        ball_detector_config=config.BALL_DETECTION_CONFIG,
        temporal_smoothing=config.BLADE_CONFIG.get('temporal_smoothing', True),
        blade_model_path=config.BLADE_CONFIG.get('model_path'),
        smplx_path=config.BLADE_CONFIG.get('smplx_path'),
        device=config.BLADE_CONFIG.get('device', 'cuda'),
        batch_size=config.BLADE_CONFIG.get('batch_size', 1),
        workers_per_gpu=config.BLADE_CONFIG.get('workers_per_gpu', 2),
        temp_output_dir=config.BLADE_CONFIG.get('temp_output_dir'),
        blade_repo_path=config.BLADE_CONFIG.get('blade_repo_path'),
        cfg_path=config.BLADE_CONFIG.get('cfg_path'),
        max_frames=config.BLADE_CONFIG.get('max_frames'),
        resize_width=config.BLADE_CONFIG.get('resize_width'),
        resize_height=config.BLADE_CONFIG.get('resize_height')
    )

    pose_frames = extractor.extract_from_video(video_path)

    if len(pose_frames) == 0:
        print("\n오류: 영상에서 포즈를 감지할 수 없습니다.")
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

    # 4. 터치 감지
    print(f"\n{'='*70}")
    print("3단계: 터치 감지 중...")
    print(f"{'='*70}\n")

    touch_detector = TouchDetector(
        velocity_threshold=config.TOUCH_DETECTION_CONFIG['velocity_threshold'],
        distance_threshold=config.TOUCH_DETECTION_CONFIG['distance_threshold'],
        peak_prominence=config.TOUCH_DETECTION_CONFIG['peak_prominence'],
        min_distance_between_touches=config.TOUCH_DETECTION_CONFIG['min_distance_between_touches'],
        merge_window=config.TOUCH_DETECTION_CONFIG['merge_window']
    )
    touches = touch_detector.detect_touches(pose_frames, ball_motion_data)

    print(f"터치 감지 완료: {len(touches)}회")
    for i, touch in enumerate(touches):
        print(f"  터치 {i+1}: 프레임 {touch.frame_number}, "
              f"방법: {touch.touch_type}, 신뢰도: {touch.confidence:.2f}")

    # 5. 드리블 종합 평가 (4가지 기준)
    print(f"\n{'='*70}")
    print("4단계: 드리블 종합 평가 중...")
    print(f"{'='*70}\n")

    dribble_scorer = DribbleScorer(
        weights=config.SCORING_CONFIG['weights'],
        head_up_config=config.SCORING_CONFIG['head_up'],
        trunk_lean_config=config.SCORING_CONFIG['trunk_lean'],
        knee_updown_config=config.SCORING_CONFIG['knee_updown'],
        alignment_config=config.SCORING_CONFIG['alignment']
    )

    dribble_result = dribble_scorer.score(pose_frames, touches, ball_motion_data)
    print(dribble_result)

    # 개선 제안
    print("\n개선 제안:")
    suggestions = dribble_scorer.get_improvement_suggestions(dribble_result)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")

    # 6. 그래프 생성
    print(f"\n{'='*70}")
    print("5단계: 그래프 생성 중...")
    print(f"{'='*70}\n")

    graphs_dir = Path(config.OUTPUT_CONFIG['graphs_dir'])
    graphs_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem

    if ball_motion_data:
        plotter = BallMotionPlotter()
        plotter.plot_motion(ball_motion_data, save_path=str(graphs_dir / f'ball_motion_{video_name}.png'))

    # 7. 스켈레톤 비디오 생성
    print(f"\n{'='*70}")
    print("6단계: 스켈레톤 비디오 생성 중...")
    print(f"{'='*70}\n")

    create_skeleton_video(video_path, pose_frames, ball_motion_data, dribble_result, video_name)

    # 8. 최종 결과 요약
    print(f"\n{'='*70}")
    print("분석 완료!")
    print(f"{'='*70}")
    print(f"\n종합 점수: {dribble_result.total_score:.1f}/100 ({dribble_result.grade})")
    print(f"\n출력 파일:")
    print(f"  - 그래프: {graphs_dir}/")
    print(f"  - 비디오: {config.OUTPUT_CONFIG['videos_dir']}/skeleton_output_{video_name}.mp4")

    print()


def create_skeleton_video(video_path: str, pose_frames, ball_motion_data=None,
                          dribble_result=None, video_name: str = "output"):
    """
    스켈레톤 비디오 생성 (공 위치, 터치, 점수 표시 포함)
    """
    videos_dir = Path(config.OUTPUT_CONFIG['videos_dir'])
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 스켈레톤 드로워 초기화
    drawer = SkeletonDrawer(color=config.OUTPUT_CONFIG['skeleton_color'])

    # 비디오 정보
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 비디오 설정
    output_path = str(videos_dir / f'skeleton_output_{video_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("스켈레톤 그리는 중...")

    # pose_frames를 frame_number로 인덱싱
    pose_dict = {pf.frame_number: pf for pf in pose_frames}

    # 터치 프레임 세트
    touch_frames_set = set(ball_motion_data.touch_frames) if ball_motion_data else set()

    # 비디오 재시작
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 프레임에 해당하는 포즈가 있으면 스켈레톤 그리기
        if frame_idx in pose_dict:
            pose_frame = pose_dict[frame_idx]

            # SMPL-X joints_2d를 SkeletonDrawer 형식으로 변환
            landmarks = convert_smplx_to_skeleton(pose_frame.joints_2d)

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

        # 4. 점수 표시 (상단)
        if dribble_result:
            score_text = f"Score: {dribble_result.total_score:.1f}/100 ({dribble_result.grade})"
            cv2.putText(frame, score_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    print(f"스켈레톤 비디오 저장: {output_path}")
    print(f"   총 {frame_idx}개 프레임 중 {len(pose_frames)}개 프레임에서 포즈 감지")


def convert_smplx_to_skeleton(joints_2d: np.ndarray) -> np.ndarray:
    """
    SMPL-X 2D 좌표를 SkeletonDrawer 형식(33 랜드마크)으로 변환
    """
    landmarks = np.zeros((33, 3))

    # SMPL-X → 33 랜드마크 매핑
    landmarks[0] = [joints_2d[15, 0], joints_2d[15, 1], 0]   # head → nose
    landmarks[2] = [joints_2d[23, 0], joints_2d[23, 1], 0]   # left_eye
    landmarks[5] = [joints_2d[24, 0], joints_2d[24, 1], 0]   # right_eye
    landmarks[11] = [joints_2d[16, 0], joints_2d[16, 1], 0]  # left_shoulder
    landmarks[12] = [joints_2d[17, 0], joints_2d[17, 1], 0]  # right_shoulder
    landmarks[13] = [joints_2d[18, 0], joints_2d[18, 1], 0]  # left_elbow
    landmarks[14] = [joints_2d[19, 0], joints_2d[19, 1], 0]  # right_elbow
    landmarks[15] = [joints_2d[20, 0], joints_2d[20, 1], 0]  # left_wrist
    landmarks[16] = [joints_2d[21, 0], joints_2d[21, 1], 0]  # right_wrist
    landmarks[23] = [joints_2d[1, 0], joints_2d[1, 1], 0]    # left_hip
    landmarks[24] = [joints_2d[2, 0], joints_2d[2, 1], 0]    # right_hip
    landmarks[25] = [joints_2d[4, 0], joints_2d[4, 1], 0]    # left_knee
    landmarks[26] = [joints_2d[5, 0], joints_2d[5, 1], 0]    # right_knee
    landmarks[27] = [joints_2d[7, 0], joints_2d[7, 1], 0]    # left_ankle
    landmarks[28] = [joints_2d[8, 0], joints_2d[8, 1], 0]    # right_ankle
    landmarks[31] = [joints_2d[10, 0], joints_2d[10, 1], 0]  # left_foot
    landmarks[32] = [joints_2d[11, 0], joints_2d[11, 1], 0]  # right_foot

    return landmarks


if __name__ == "__main__":
    main()
