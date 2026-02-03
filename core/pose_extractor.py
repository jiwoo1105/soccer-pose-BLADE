# core/pose_extractor.py
"""
=============================================================================
BLADE를 사용한 3D 포즈 추출
=============================================================================

BLADE (Body Landmarks Anatomical Dense Estimation)를 사용하여
비디오에서 정확한 SMPL-X 54개 관절 3D 포즈를 추출합니다.

주요 기능:
1. 비디오에서 프레임별 3D 포즈 추출
2. 골반 중심 좌표계로 뷰 불변성 보장
3. 공 탐지 통합 (YOLOv9 + 스켈레톤 폴백)
"""

import cv2
import numpy as np
from typing import List, Optional

from .blade_wrapper import BladeWrapper, PoseFrame3D, SMPLX_JOINT_INDICES

try:
    from .ball_detector import HybridBallDetector
    BALL_DETECTION_AVAILABLE = True
except ImportError:
    BALL_DETECTION_AVAILABLE = False
    HybridBallDetector = None


class PoseExtractor:
    """
    비디오에서 SMPL-X 3D 포즈를 추출하는 클래스

    BLADE를 사용하여 54개 관절의 정확한 3D 좌표를 추출합니다.

    사용 예시:
        >>> extractor = PoseExtractor()
        >>> pose_frames = extractor.extract_from_video("soccer.mp4")
        >>> print(f"추출된 프레임: {len(pose_frames)}")
    """

    def __init__(self,
                 detect_ball: bool = True,
                 ball_detector_config: Optional[dict] = None,
                 temporal_smoothing: bool = True,
                 blade_model_path: Optional[str] = None,
                 smplx_path: Optional[str] = None,
                 device: str = 'cuda',
                 batch_size: int = 1,
                 workers_per_gpu: int = 2,
                 temp_output_dir: Optional[str] = None,
                 blade_repo_path: Optional[str] = None,
                 cfg_path: Optional[str] = None,
                 max_frames: Optional[int] = None,
                 resize_width: Optional[int] = None,
                 resize_height: Optional[int] = None,
                 **kwargs):
        """
        PoseExtractor 초기화

        Args:
            detect_ball: 공 탐지 활성화 여부
            ball_detector_config: 공 탐지 설정
            temporal_smoothing: 시간적 스무딩
            blade_model_path: BLADE 모델 경로
            smplx_path: SMPL-X 모델 경로
            device: 연산 장치 ('cuda' 또는 'cpu')
            batch_size: BLADE 처리 배치 크기
            workers_per_gpu: 데이터 로더 워커 수
            temp_output_dir: BLADE 중간 결과 저장 경로
            blade_repo_path: BLADE 저장소 루트 경로
            cfg_path: BLADE 설정 파일 경로
            max_frames: 최대 처리 프레임 수 (스모크 테스트용)
            resize_width: 입력 리사이즈 너비 (None이면 원본 유지)
            resize_height: 입력 리사이즈 높이 (None이면 원본 유지)
        """
        self.detect_ball = detect_ball
        self.ball_detector = None

        # BLADE 래퍼 초기화
        self.blade_wrapper = BladeWrapper(
            model_path=blade_model_path,
            smplx_path=smplx_path,
            device=device,
            temporal_smoothing=temporal_smoothing,
            batch_size=batch_size,
            workers_per_gpu=workers_per_gpu,
            temp_output_dir=temp_output_dir,
            blade_repo_path=blade_repo_path,
            cfg_path=cfg_path,
            max_frames=max_frames,
            resize_width=resize_width,
            resize_height=resize_height
        )

        # 공 탐지기 초기화
        if detect_ball and BALL_DETECTION_AVAILABLE:
            if ball_detector_config:
                self.ball_detector = HybridBallDetector(**ball_detector_config)
            else:
                self.ball_detector = HybridBallDetector()
        elif detect_ball:
            print("Warning: 공 탐지 사용 불가. ultralytics를 설치하세요.")
            self.detect_ball = False

    def extract_from_video(self, video_path: str) -> List[PoseFrame3D]:
        """
        비디오에서 모든 프레임의 3D 포즈 추출

        Args:
            video_path: 비디오 파일 경로

        Returns:
            List[PoseFrame3D]: 포즈가 감지된 프레임들의 리스트
        """
        return self.blade_wrapper.extract_from_video(
            video_path,
            ball_detector=self.ball_detector
        )

    def extract_from_frame(self, frame: np.ndarray) -> Optional[PoseFrame3D]:
        """
        단일 프레임에서 포즈 추출

        Args:
            frame: BGR 이미지 (OpenCV 형식)

        Returns:
            PoseFrame3D 또는 None
        """
        return self.blade_wrapper._extract_single_frame(
            frame, 0, 30.0,
            frame.shape[1], frame.shape[0],
            self.ball_detector
        )
