# analysis/trunk_pose_analyzer.py
"""상체 자세 분석 모듈 - 무릎-엉덩이-어깨 각도 계산"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from utils.math_utils import calculate_angle


@dataclass
class TrunkPoseData:
    """상체 자세 분석 결과"""
    frame_numbers: np.ndarray           # 분석된 프레임 번호들
    trunk_angles: np.ndarray            # 각 프레임의 상체 각도 (degrees)
    mean_angle: float                   # 평균 상체 각도

    def __str__(self):
        """사용자 친화적인 출력 포맷"""
        result = []
        result.append("="*70)
        result.append("상체 자세 분석 결과")
        result.append("="*70)
        result.append(f"총 분석 프레임 수: {len(self.frame_numbers)}")
        result.append(f"평균 상체 각도: {self.mean_angle:.2f}°")
        result.append("="*70)
        return "\n".join(result)


class TrunkPoseAnalyzer:
    """상체 자세 분석 - 무릎-엉덩이-어깨 각도 계산"""

    # 랜드마크 인덱스
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26

    def __init__(self, min_visibility_threshold: float = 0.5):
        """
        Args:
            min_visibility_threshold: 최소 랜드마크 신뢰도 (0.5 권장)
        """
        self.min_visibility_threshold = min_visibility_threshold

    def calculate_trunk_angle(self, pose_frame) -> Optional[float]:
        """
        단일 프레임에서 상체 각도 계산

        Args:
            pose_frame: PoseFrame 객체 (world_landmarks 사용)

        Returns:
            상체 각도 (degrees) 또는 None (신뢰도 낮을 경우)

        계산 방법:
            1. 왼쪽 각도 = 무릎(LEFT_KNEE) - 엉덩이(LEFT_HIP) - 어깨(LEFT_SHOULDER)
            2. 오른쪽 각도 = 무릎(RIGHT_KNEE) - 엉덩이(RIGHT_HIP) - 어깨(RIGHT_SHOULDER)
            3. 상체 각도 = (왼쪽 각도 + 오른쪽 각도) / 2
        """
        # Visibility 체크
        required_landmarks = [
            self.LEFT_KNEE, self.LEFT_HIP, self.LEFT_SHOULDER,
            self.RIGHT_KNEE, self.RIGHT_HIP, self.RIGHT_SHOULDER
        ]
        visibilities = [pose_frame.visibility[idx] for idx in required_landmarks]

        if any(v < self.min_visibility_threshold for v in visibilities):
            return None

        # world_landmarks 사용 (3D 실제 좌표)
        world_landmarks = pose_frame.world_landmarks

        # 왼쪽 각도: 무릎 - 엉덩이 - 어깨 (엉덩이가 꼭지점)
        left_knee = world_landmarks[self.LEFT_KNEE]
        left_hip = world_landmarks[self.LEFT_HIP]
        left_shoulder = world_landmarks[self.LEFT_SHOULDER]
        left_angle = calculate_angle(left_knee, left_hip, left_shoulder)

        # 오른쪽 각도: 무릎 - 엉덩이 - 어깨 (엉덩이가 꼭지점)
        right_knee = world_landmarks[self.RIGHT_KNEE]
        right_hip = world_landmarks[self.RIGHT_HIP]
        right_shoulder = world_landmarks[self.RIGHT_SHOULDER]
        right_angle = calculate_angle(right_knee, right_hip, right_shoulder)

        # 양쪽 평균
        trunk_angle = (left_angle + right_angle) / 2

        return trunk_angle

    def analyze(self, pose_frames):
        """
        PoseFrame 리스트에서 상체 자세 분석

        Args:
            pose_frames: PoseFrame 객체 리스트

        Returns:
            TrunkPoseData 객체 또는 None (유효한 프레임이 없을 경우)
        """
        if len(pose_frames) == 0:
            return None

        # 각 프레임의 상체 각도 계산
        valid_data = []
        for pf in pose_frames:
            angle = self.calculate_trunk_angle(pf)
            if angle is not None:
                valid_data.append({
                    'frame_number': pf.frame_number,
                    'angle': angle
                })

        if len(valid_data) == 0:
            return None

        # 데이터 추출
        frame_numbers = np.array([d['frame_number'] for d in valid_data])
        trunk_angles = np.array([d['angle'] for d in valid_data])

        # 평균 계산
        mean_angle = np.mean(trunk_angles)

        return TrunkPoseData(
            frame_numbers=frame_numbers,
            trunk_angles=trunk_angles,
            mean_angle=mean_angle
        )
