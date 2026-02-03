# scoring/head_up_scorer.py
"""
=============================================================================
헤드업 점수 평가 모듈
=============================================================================

터치 순간에 헤드업 상태를 평가합니다.

규칙:
- 터치 순간 (±5 프레임): 시선이 수평에 가까울수록 높은 점수
- 터치 전후: 공을 봐도 패널티 없음 (10점 유지)

시선 방향 계산:
- 눈 중심 → 턱 방향 벡터
- 수평(0도)에 가까울수록 좋음

점수 기준:
- 터치 순간 수평 ±10도: 10점
- 벗어날수록 감점 (3도당 1점)
- 30도 이상 벗어나면: 0점
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class HeadUpScore:
    """헤드업 점수 데이터"""
    total_score: float  # 0-10 점수
    frame_scores: np.ndarray  # 각 프레임별 점수
    touch_scores: List[float]  # 각 터치 순간별 점수
    mean_gaze_angle: float  # 평균 시선 각도
    touch_gaze_angles: List[float]  # 터치 순간 시선 각도들


class HeadUpScorer:
    """
    헤드업 점수 평가기

    터치 순간에 고개를 들고 있는지 평가합니다.
    3D 시선 방향을 계산하여 뷰 불변적인 점수를 산출합니다.

    사용 예시:
        >>> scorer = HeadUpScorer()
        >>> score = scorer.score(pose_frames, touches)
        >>> print(f"헤드업 점수: {score.total_score:.1f}/10")
    """

    def __init__(self,
                 touch_window: int = 5,
                 optimal_angle_range: Tuple[float, float] = (-10, 10),
                 worst_angle: float = 30.0,
                 min_visibility: float = 0.5):
        """
        HeadUpScorer 초기화

        Args:
            touch_window: 터치 전후 평가 프레임 수
            optimal_angle_range: 최적 시선 각도 범위 (수평 기준, 도)
            worst_angle: 최악 시선 각도 (이 이상이면 0점)
            min_visibility: 최소 관절 신뢰도
        """
        self.touch_window = touch_window
        self.optimal_min = optimal_angle_range[0]
        self.optimal_max = optimal_angle_range[1]
        self.worst_angle = worst_angle
        self.min_visibility = min_visibility

    def score(self, pose_frames: List, touches: List) -> HeadUpScore:
        """
        포즈 프레임과 터치 이벤트로 헤드업 점수 계산

        Args:
            pose_frames: PoseFrame3D 또는 PoseFrame 리스트
            touches: TouchEvent 리스트 또는 터치 프레임 번호 리스트

        Returns:
            HeadUpScore: 점수 데이터
        """
        if not pose_frames:
            return HeadUpScore(
                total_score=0.0,
                frame_scores=np.array([]),
                touch_scores=[],
                mean_gaze_angle=0.0,
                touch_gaze_angles=[]
            )

        # 터치 프레임 번호 추출
        if touches and hasattr(touches[0], 'frame_number'):
            touch_frames = [t.frame_number for t in touches]
        else:
            touch_frames = list(touches) if touches else []

        # 각 프레임별 시선 각도 및 점수 계산
        frame_angles = []
        frame_scores = []

        for pf in pose_frames:
            gaze_angle = self._calculate_gaze_angle(pf)
            if gaze_angle is not None:
                frame_angles.append(gaze_angle)

                # 터치 순간인지 확인
                is_touch_moment = any(
                    abs(pf.frame_number - tf) <= self.touch_window
                    for tf in touch_frames
                )

                if is_touch_moment:
                    # 터치 순간: 수평에 가까울수록 높은 점수
                    score = self._score_touch_moment(gaze_angle)
                else:
                    # 터치 전후: 패널티 없음
                    score = 10.0

                frame_scores.append(score)
            else:
                frame_angles.append(np.nan)
                frame_scores.append(np.nan)

        # 터치 순간별 점수 추출
        touch_scores = []
        touch_gaze_angles = []

        for tf in touch_frames:
            # 터치 프레임에 가장 가까운 포즈 프레임 찾기
            closest_idx = self._find_closest_frame_idx(pose_frames, tf)
            if closest_idx is not None and not np.isnan(frame_angles[closest_idx]):
                touch_gaze_angles.append(frame_angles[closest_idx])
                touch_scores.append(frame_scores[closest_idx])

        # 총점 계산 (터치 순간만 평균)
        valid_touch_scores = [s for s in touch_scores if not np.isnan(s)]
        if valid_touch_scores:
            total_score = np.mean(valid_touch_scores)
        else:
            # 터치가 없으면 전체 프레임 평균
            valid_scores = [s for s in frame_scores if not np.isnan(s)]
            total_score = np.mean(valid_scores) if valid_scores else 0.0

        # 평균 시선 각도
        valid_angles = [a for a in frame_angles if not np.isnan(a)]
        mean_gaze_angle = np.mean(valid_angles) if valid_angles else 0.0

        return HeadUpScore(
            total_score=float(total_score),
            frame_scores=np.array(frame_scores),
            touch_scores=touch_scores,
            mean_gaze_angle=float(mean_gaze_angle),
            touch_gaze_angles=touch_gaze_angles
        )

    def _calculate_gaze_angle(self, pose_frame) -> Optional[float]:
        """
        시선 각도 계산

        눈 중심에서 턱 방향으로의 벡터를 계산하고,
        이 벡터가 수평면과 이루는 각도를 반환합니다.

        Returns:
            float: 시선 각도 (도, 아래쪽이 양수)
        """
        # PoseFrame3D (SMPL-X)
        if hasattr(pose_frame, 'joints_3d') and pose_frame.joints_3d is not None:
            joints = pose_frame.joints_3d
            confidence = pose_frame.confidence

            # SMPL-X: left_eye=23, right_eye=24, jaw=22, head=15
            if confidence[23] < self.min_visibility or confidence[24] < self.min_visibility:
                return None

            eye_center = (joints[23] + joints[24]) / 2
            jaw = joints[22] if confidence[22] >= self.min_visibility else joints[15]

            # 시선 방향: 눈 → 턱 (아래를 보는 방향)
            gaze_vector = jaw - eye_center

            # 수평면과의 각도 계산
            return self._angle_from_horizontal(gaze_vector)

        # PoseFrame (MediaPipe)
        elif hasattr(pose_frame, 'world_landmarks') and pose_frame.world_landmarks is not None:
            landmarks = pose_frame.world_landmarks
            visibility = pose_frame.visibility

            # MediaPipe: LEFT_EYE=2, RIGHT_EYE=5, NOSE=0
            if visibility[2] < self.min_visibility or visibility[5] < self.min_visibility:
                return None

            eye_center = (landmarks[2] + landmarks[5]) / 2
            nose = landmarks[0]

            # 시선 방향 추정: 눈 중심 → 코
            gaze_vector = nose - eye_center

            return self._angle_from_horizontal(gaze_vector)

        return None

    def _angle_from_horizontal(self, vector: np.ndarray) -> float:
        """벡터와 수평면 사이의 각도 (도)"""
        # 3D 벡터에서 수직 성분의 비율로 각도 계산
        if len(vector) < 3:
            return 0.0

        horizontal_length = np.sqrt(vector[0]**2 + vector[2]**2)
        vertical = abs(vector[1])  # Y축 (상하)

        if horizontal_length < 1e-10:
            return 90.0 if vector[1] > 0 else -90.0

        angle_rad = np.arctan2(vertical, horizontal_length)
        angle_deg = np.degrees(angle_rad)

        # 아래를 보면 양수, 위를 보면 음수
        if vector[1] > 0:  # MediaPipe에서 Y축 아래가 양수
            return angle_deg
        else:
            return -angle_deg

    def _score_touch_moment(self, gaze_angle: float) -> float:
        """
        터치 순간의 시선 각도에 대한 점수

        Args:
            gaze_angle: 시선 각도 (도, 아래가 양수)

        Returns:
            float: 0-10 점수
        """
        # 최적 범위 내: 10점
        if self.optimal_min <= gaze_angle <= self.optimal_max:
            return 10.0

        # 범위 벗어난 정도 계산
        if gaze_angle < self.optimal_min:
            deviation = self.optimal_min - gaze_angle
        else:
            deviation = gaze_angle - self.optimal_max

        # 최악 각도 이상: 0점
        if deviation >= self.worst_angle:
            return 0.0

        # 선형 감점 (3도당 1점)
        penalty = deviation / 3.0
        return max(0.0, 10.0 - penalty)

    def _find_closest_frame_idx(self, pose_frames: List, target_frame: int) -> Optional[int]:
        """목표 프레임 번호에 가장 가까운 포즈 프레임 인덱스 찾기"""
        min_dist = float('inf')
        closest_idx = None

        for i, pf in enumerate(pose_frames):
            dist = abs(pf.frame_number - target_frame)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def get_phase_score(self, pose_frame, phase: str) -> float:
        """
        특정 단계의 헤드업 점수

        Args:
            pose_frame: 포즈 프레임
            phase: 'before', 'touch', 'after'

        Returns:
            float: 0-10 점수
        """
        gaze_angle = self._calculate_gaze_angle(pose_frame)
        if gaze_angle is None:
            return 10.0  # 측정 불가시 패널티 없음

        if phase == 'touch':
            return self._score_touch_moment(gaze_angle)
        else:
            # 터치 전후는 패널티 없음
            return 10.0
