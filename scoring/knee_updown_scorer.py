# scoring/knee_updown_scorer.py
"""
=============================================================================
무릎 업다운(Knee Up-Down) 점수 평가 모듈
=============================================================================

터치 순간 무릎의 굽힘-펴짐 패턴을 평가합니다.

규칙:
- 터치 순간: 양 무릎이 굽힘 → 최저점 → 펴짐 패턴을 보여야 함
- 양쪽 무릎 모두 패턴이 있으면 10점
- 대칭성 보너스 추가

무릎 각도 계산:
- hip - knee - ankle 세 점으로 무릎 각도 계산
- 완전히 펴면 180도, 굽히면 각도 감소

패턴 감지:
- 터치 전후 ±10프레임 윈도우
- 각도가 감소 → 최저점 → 증가 패턴 확인
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class KneeUpDownScore:
    """무릎 업다운 점수 데이터"""
    total_score: float  # 0-10 점수
    touch_scores: List[float]  # 각 터치별 점수
    left_patterns: List[bool]  # 각 터치별 왼쪽 무릎 패턴 존재
    right_patterns: List[bool]  # 각 터치별 오른쪽 무릎 패턴 존재
    symmetry_scores: List[float]  # 각 터치별 대칭성 점수
    mean_left_angle: float  # 평균 왼쪽 무릎 각도
    mean_right_angle: float  # 평균 오른쪽 무릎 각도


class KneeUpDownScorer:
    """
    무릎 업다운 점수 평가기

    터치 순간 무릎의 굽힘-펴짐 패턴을 감지하고 평가합니다.
    3D 좌표를 사용하여 뷰 불변적인 점수를 산출합니다.

    사용 예시:
        >>> scorer = KneeUpDownScorer()
        >>> score = scorer.score(pose_frames, touches)
        >>> print(f"무릎 업다운 점수: {score.total_score:.1f}/10")
    """

    def __init__(self,
                 touch_window: int = 10,
                 min_angle_change: float = 10.0,
                 symmetry_bonus: float = 1.0,
                 min_visibility: float = 0.5):
        """
        KneeUpDownScorer 초기화

        Args:
            touch_window: 터치 전후 분석 프레임 수
            min_angle_change: 패턴으로 인정할 최소 각도 변화 (도)
            symmetry_bonus: 양쪽 대칭시 추가 점수
            min_visibility: 최소 관절 신뢰도
        """
        self.touch_window = touch_window
        self.min_angle_change = min_angle_change
        self.symmetry_bonus = symmetry_bonus
        self.min_visibility = min_visibility

    def score(self, pose_frames: List, touches: List) -> KneeUpDownScore:
        """
        포즈 프레임과 터치 이벤트로 무릎 업다운 점수 계산

        Args:
            pose_frames: PoseFrame3D 또는 PoseFrame 리스트
            touches: TouchEvent 리스트 또는 터치 프레임 번호 리스트

        Returns:
            KneeUpDownScore: 점수 데이터
        """
        if not pose_frames or not touches:
            return KneeUpDownScore(
                total_score=0.0,
                touch_scores=[],
                left_patterns=[],
                right_patterns=[],
                symmetry_scores=[],
                mean_left_angle=0.0,
                mean_right_angle=0.0
            )

        # 터치 프레임 번호 추출
        if hasattr(touches[0], 'frame_number'):
            touch_frames = [t.frame_number for t in touches]
        else:
            touch_frames = list(touches)

        # 프레임 번호 → 인덱스 매핑
        frame_to_idx = {pf.frame_number: i for i, pf in enumerate(pose_frames)}

        # 각 프레임별 무릎 각도 계산
        left_angles = []
        right_angles = []

        for pf in pose_frames:
            left_angle = self._calculate_knee_angle(pf, 'left')
            right_angle = self._calculate_knee_angle(pf, 'right')
            left_angles.append(left_angle if left_angle is not None else np.nan)
            right_angles.append(right_angle if right_angle is not None else np.nan)

        left_angles = np.array(left_angles)
        right_angles = np.array(right_angles)

        # 각 터치별 점수 계산
        touch_scores = []
        left_patterns = []
        right_patterns = []
        symmetry_scores = []

        for tf in touch_frames:
            # 터치 주변 인덱스 찾기
            center_idx = frame_to_idx.get(tf)
            if center_idx is None:
                # 가장 가까운 프레임 찾기
                center_idx = self._find_closest_idx(pose_frames, tf)

            if center_idx is None:
                touch_scores.append(0.0)
                left_patterns.append(False)
                right_patterns.append(False)
                symmetry_scores.append(0.0)
                continue

            # 윈도우 범위
            start_idx = max(0, center_idx - self.touch_window)
            end_idx = min(len(pose_frames), center_idx + self.touch_window + 1)

            # 왼쪽 무릎 패턴 감지
            left_window = left_angles[start_idx:end_idx]
            left_has_pattern = self._detect_flex_extend_pattern(left_window)

            # 오른쪽 무릎 패턴 감지
            right_window = right_angles[start_idx:end_idx]
            right_has_pattern = self._detect_flex_extend_pattern(right_window)

            # 점수 계산
            base_score = 0.0
            if left_has_pattern and right_has_pattern:
                base_score = 8.0
            elif left_has_pattern or right_has_pattern:
                base_score = 5.0

            # 대칭성 점수
            symmetry = self._calculate_symmetry(left_window, right_window)
            symmetry_score = symmetry * self.symmetry_bonus

            total = min(10.0, base_score + symmetry_score)

            touch_scores.append(total)
            left_patterns.append(left_has_pattern)
            right_patterns.append(right_has_pattern)
            symmetry_scores.append(symmetry_score)

        # 전체 평균
        total_score = np.mean(touch_scores) if touch_scores else 0.0

        # 평균 무릎 각도
        valid_left = left_angles[~np.isnan(left_angles)]
        valid_right = right_angles[~np.isnan(right_angles)]

        return KneeUpDownScore(
            total_score=float(total_score),
            touch_scores=touch_scores,
            left_patterns=left_patterns,
            right_patterns=right_patterns,
            symmetry_scores=symmetry_scores,
            mean_left_angle=float(np.mean(valid_left)) if len(valid_left) > 0 else 0.0,
            mean_right_angle=float(np.mean(valid_right)) if len(valid_right) > 0 else 0.0
        )

    def _calculate_knee_angle(self, pose_frame, side: str) -> Optional[float]:
        """
        무릎 각도 계산

        hip - knee - ankle 세 점으로 무릎 각도를 계산합니다.
        완전히 펴면 180도, 굽히면 각도가 감소합니다.

        Args:
            pose_frame: 포즈 프레임
            side: 'left' 또는 'right'

        Returns:
            float: 무릎 각도 (도)
        """
        # PoseFrame3D (SMPL-X)
        if hasattr(pose_frame, 'joints_3d') and pose_frame.joints_3d is not None:
            joints = pose_frame.joints_3d
            confidence = pose_frame.confidence

            if side == 'left':
                # SMPL-X: left_hip=1, left_knee=4, left_ankle=7
                hip_idx, knee_idx, ankle_idx = 1, 4, 7
            else:
                # SMPL-X: right_hip=2, right_knee=5, right_ankle=8
                hip_idx, knee_idx, ankle_idx = 2, 5, 8

            if (confidence[hip_idx] < self.min_visibility or
                confidence[knee_idx] < self.min_visibility or
                confidence[ankle_idx] < self.min_visibility):
                return None

            hip = joints[hip_idx]
            knee = joints[knee_idx]
            ankle = joints[ankle_idx]

            return self._calculate_angle_3points(hip, knee, ankle)

        # PoseFrame (MediaPipe)
        elif hasattr(pose_frame, 'world_landmarks') and pose_frame.world_landmarks is not None:
            landmarks = pose_frame.world_landmarks
            visibility = pose_frame.visibility

            if side == 'left':
                # MediaPipe: LEFT_HIP=23, LEFT_KNEE=25, LEFT_ANKLE=27
                hip_idx, knee_idx, ankle_idx = 23, 25, 27
            else:
                # MediaPipe: RIGHT_HIP=24, RIGHT_KNEE=26, RIGHT_ANKLE=28
                hip_idx, knee_idx, ankle_idx = 24, 26, 28

            if (visibility[hip_idx] < self.min_visibility or
                visibility[knee_idx] < self.min_visibility or
                visibility[ankle_idx] < self.min_visibility):
                return None

            hip = landmarks[hip_idx]
            knee = landmarks[knee_idx]
            ankle = landmarks[ankle_idx]

            return self._calculate_angle_3points(hip, knee, ankle)

        return None

    def _calculate_angle_3points(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """세 점으로 p2에서의 각도 계산"""
        v1 = p1 - p2
        v2 = p3 - p2

        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return 180.0

        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)

        return np.degrees(angle_rad)

    def _detect_flex_extend_pattern(self, angles: np.ndarray) -> bool:
        """
        굽힘-펴짐 패턴 감지

        각도가 감소 → 최저점 → 증가하는 패턴을 찾습니다.

        Args:
            angles: 시간순 무릎 각도 배열

        Returns:
            bool: 패턴 존재 여부
        """
        valid_angles = angles[~np.isnan(angles)]
        if len(valid_angles) < 5:
            return False

        # 최솟값 인덱스 찾기
        min_idx = np.argmin(valid_angles)

        # 최솟값이 양 끝이 아니어야 함
        if min_idx == 0 or min_idx == len(valid_angles) - 1:
            return False

        # 최솟값 전후의 각도 변화 확인
        before_min = valid_angles[:min_idx]
        after_min = valid_angles[min_idx+1:]

        # 굽힘: 전반부에서 각도 감소
        flex_change = np.max(before_min) - valid_angles[min_idx]

        # 펴짐: 후반부에서 각도 증가
        extend_change = np.max(after_min) - valid_angles[min_idx]

        # 두 방향 모두 최소 변화량 이상이어야 패턴
        return flex_change >= self.min_angle_change and extend_change >= self.min_angle_change

    def _calculate_symmetry(self, left_angles: np.ndarray, right_angles: np.ndarray) -> float:
        """
        양쪽 무릎 대칭성 계산

        두 무릎의 각도 변화 패턴이 얼마나 유사한지 평가합니다.

        Returns:
            float: 0-1 대칭성 점수
        """
        # NaN 제거
        valid_mask = ~np.isnan(left_angles) & ~np.isnan(right_angles)
        if np.sum(valid_mask) < 3:
            return 0.0

        left_valid = left_angles[valid_mask]
        right_valid = right_angles[valid_mask]

        # 상관계수로 대칭성 평가
        if np.std(left_valid) < 1e-10 or np.std(right_valid) < 1e-10:
            return 0.5  # 변화가 거의 없으면 중간 점수

        correlation = np.corrcoef(left_valid, right_valid)[0, 1]

        # 상관계수를 0-1 범위로 변환 (음의 상관도 고려)
        symmetry = (correlation + 1) / 2

        return float(symmetry)

    def _find_closest_idx(self, pose_frames: List, target_frame: int) -> Optional[int]:
        """목표 프레임에 가장 가까운 인덱스 찾기"""
        min_dist = float('inf')
        closest_idx = None

        for i, pf in enumerate(pose_frames):
            dist = abs(pf.frame_number - target_frame)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def get_knee_angles(self, pose_frames: List) -> Tuple[np.ndarray, np.ndarray]:
        """모든 프레임의 양쪽 무릎 각도 반환"""
        left_angles = []
        right_angles = []

        for pf in pose_frames:
            left_angles.append(self._calculate_knee_angle(pf, 'left') or np.nan)
            right_angles.append(self._calculate_knee_angle(pf, 'right') or np.nan)

        return np.array(left_angles), np.array(right_angles)
