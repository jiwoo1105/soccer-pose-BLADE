# scoring/trunk_lean_scorer.py
"""
=============================================================================
상체 기울기(Trunk Lean) 점수 평가 모듈
=============================================================================

전체 동작 중 상체가 적절히 앞으로 숙여져 있는지 평가합니다.

규칙:
- 상체가 수직에서 70-85도 앞으로 기울어진 상태가 최적
- 너무 서있으면(<60도) 감점
- 너무 숙이면(>90도) 감점

상체 각도 계산:
- spine3(상체 상단) - pelvis(골반) 벡터
- 수직축(Y축)과의 각도

점수 기준:
- 70-85도: 10점
- 범위 벗어날수록 감점
- 60도 미만 또는 90도 초과: 0점
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrunkLeanScore:
    """상체 기울기 점수 데이터"""
    total_score: float  # 0-10 점수
    frame_scores: np.ndarray  # 각 프레임별 점수
    frame_angles: np.ndarray  # 각 프레임별 상체 각도
    mean_angle: float  # 평균 상체 각도
    angle_std: float  # 상체 각도 표준편차


class TrunkLeanScorer:
    """
    상체 기울기 점수 평가기

    드리블 중 상체가 적절히 앞으로 기울어져 있는지 평가합니다.
    3D 좌표를 사용하여 뷰 불변적인 점수를 산출합니다.

    사용 예시:
        >>> scorer = TrunkLeanScorer()
        >>> score = scorer.score(pose_frames)
        >>> print(f"상체 기울기 점수: {score.total_score:.1f}/10")
    """

    def __init__(self,
                 optimal_range: Tuple[float, float] = (70, 85),
                 worst_low: float = 60,
                 worst_high: float = 90,
                 min_visibility: float = 0.5):
        """
        TrunkLeanScorer 초기화

        Args:
            optimal_range: 최적 상체 각도 범위 (도)
            worst_low: 최저 허용 각도 (이하면 0점)
            worst_high: 최고 허용 각도 (이상이면 0점)
            min_visibility: 최소 관절 신뢰도
        """
        self.optimal_min = optimal_range[0]
        self.optimal_max = optimal_range[1]
        self.worst_low = worst_low
        self.worst_high = worst_high
        self.min_visibility = min_visibility

    def score(self, pose_frames: List) -> TrunkLeanScore:
        """
        포즈 프레임으로 상체 기울기 점수 계산

        Args:
            pose_frames: PoseFrame3D 또는 PoseFrame 리스트

        Returns:
            TrunkLeanScore: 점수 데이터
        """
        if not pose_frames:
            return TrunkLeanScore(
                total_score=0.0,
                frame_scores=np.array([]),
                frame_angles=np.array([]),
                mean_angle=0.0,
                angle_std=0.0
            )

        # 각 프레임별 상체 각도 및 점수 계산
        frame_angles = []
        frame_scores = []

        for pf in pose_frames:
            angle = self._calculate_trunk_angle(pf)
            if angle is not None:
                frame_angles.append(angle)
                score = self._score_angle(angle)
                frame_scores.append(score)
            else:
                frame_angles.append(np.nan)
                frame_scores.append(np.nan)

        # 유효한 값들로 통계 계산
        valid_angles = [a for a in frame_angles if not np.isnan(a)]
        valid_scores = [s for s in frame_scores if not np.isnan(s)]

        if valid_angles:
            mean_angle = np.mean(valid_angles)
            angle_std = np.std(valid_angles)
            total_score = np.mean(valid_scores)
        else:
            mean_angle = 0.0
            angle_std = 0.0
            total_score = 0.0

        return TrunkLeanScore(
            total_score=float(total_score),
            frame_scores=np.array(frame_scores),
            frame_angles=np.array(frame_angles),
            mean_angle=float(mean_angle),
            angle_std=float(angle_std)
        )

    def _calculate_trunk_angle(self, pose_frame) -> Optional[float]:
        """
        상체 기울기 각도 계산

        spine3(상체 상단)에서 pelvis(골반)까지의 벡터가
        수직축과 이루는 각도를 계산합니다.

        Returns:
            float: 상체 각도 (도, 0=수직, 90=수평)
        """
        # PoseFrame3D (SMPL-X)
        if hasattr(pose_frame, 'joints_3d') and pose_frame.joints_3d is not None:
            joints = pose_frame.joints_3d
            confidence = pose_frame.confidence

            # SMPL-X: pelvis=0, spine3=9
            if confidence[0] < self.min_visibility or confidence[9] < self.min_visibility:
                return None

            pelvis = joints[0]
            spine3 = joints[9]

            # 상체 벡터: 골반 → 상체 상단
            spine_vector = spine3 - pelvis

            return self._angle_with_vertical(spine_vector)

        # PoseFrame (MediaPipe)
        elif hasattr(pose_frame, 'world_landmarks') and pose_frame.world_landmarks is not None:
            landmarks = pose_frame.world_landmarks
            visibility = pose_frame.visibility

            # MediaPipe: LEFT_HIP=23, RIGHT_HIP=24, LEFT_SHOULDER=11, RIGHT_SHOULDER=12
            if (visibility[23] < self.min_visibility or visibility[24] < self.min_visibility or
                visibility[11] < self.min_visibility or visibility[12] < self.min_visibility):
                return None

            hip_center = (landmarks[23] + landmarks[24]) / 2
            shoulder_center = (landmarks[11] + landmarks[12]) / 2

            # 상체 벡터: 골반 중심 → 어깨 중심
            spine_vector = shoulder_center - hip_center

            return self._angle_with_vertical(spine_vector)

        return None

    def _angle_with_vertical(self, vector: np.ndarray) -> float:
        """벡터와 수직축(Y축) 사이의 각도 (도)"""
        if len(vector) < 3:
            return 0.0

        # 수직축 (위쪽 방향)
        # MediaPipe에서는 Y가 아래쪽이 양수이므로 반전
        vertical = np.array([0, -1, 0])

        # 벡터 정규화
        vector_norm = np.linalg.norm(vector)
        if vector_norm < 1e-10:
            return 0.0

        vector_unit = vector / vector_norm

        # 내적으로 각도 계산
        cos_angle = np.clip(np.dot(vector_unit, vertical), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)

        return np.degrees(angle_rad)

    def _score_angle(self, angle: float) -> float:
        """
        상체 각도에 대한 점수

        Args:
            angle: 상체 각도 (도)

        Returns:
            float: 0-10 점수
        """
        # 최적 범위 내: 10점
        if self.optimal_min <= angle <= self.optimal_max:
            return 10.0

        # 너무 서있음 (각도가 작음)
        if angle < self.optimal_min:
            if angle <= self.worst_low:
                return 0.0
            # 선형 보간
            range_size = self.optimal_min - self.worst_low
            deviation = self.optimal_min - angle
            return 10.0 * (1 - deviation / range_size)

        # 너무 숙임 (각도가 큼)
        if angle > self.optimal_max:
            if angle >= self.worst_high:
                return 0.0
            # 선형 보간
            range_size = self.worst_high - self.optimal_max
            deviation = angle - self.optimal_max
            return 10.0 * (1 - deviation / range_size)

        return 10.0

    def calculate_single_frame(self, pose_frame) -> Tuple[Optional[float], float]:
        """
        단일 프레임의 상체 각도와 점수

        Returns:
            (angle, score): 각도와 점수 튜플
        """
        angle = self._calculate_trunk_angle(pose_frame)
        if angle is None:
            return None, 0.0

        score = self._score_angle(angle)
        return angle, score
