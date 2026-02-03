# core/touch_detector.py
"""
=============================================================================
하이브리드 터치 감지 모듈
=============================================================================

공 터치 순간을 감지하는 모듈입니다.
두 가지 방법을 교차 검증하여 정확도를 높입니다:

1. 속도 기반 감지: 공의 속도가 최소가 되는 순간
2. 거리 기반 감지: 발과 공의 거리가 가까운 순간

두 방법의 결과를 병합하여 최종 터치 프레임을 결정합니다.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.signal import find_peaks


@dataclass
class TouchEvent:
    """
    터치 이벤트 데이터

    Attributes:
        frame_number: 터치가 발생한 프레임 번호
        timestamp: 터치 시간 (초)
        touch_type: 'velocity', 'distance', 또는 'both'
        foot_side: 'left' 또는 'right' (터치한 발)
        confidence: 터치 감지 신뢰도 (0-1)
        ball_position: 터치 순간 공의 위치
        foot_position: 터치 순간 발의 위치
    """
    frame_number: int
    timestamp: float
    touch_type: str
    foot_side: Optional[str] = None
    confidence: float = 1.0
    ball_position: Optional[Tuple[float, float]] = None
    foot_position: Optional[Tuple[float, float, float]] = None


class TouchDetector:
    """
    하이브리드 터치 감지기

    속도 기반과 거리 기반 감지를 결합하여
    공 터치 순간을 정확하게 감지합니다.

    사용 예시:
        >>> detector = TouchDetector()
        >>> touches = detector.detect_touches(pose_frames, ball_positions, ball_velocities)
        >>> print(f"터치 {len(touches)}회 감지")
    """

    def __init__(self,
                 velocity_threshold: float = 5.0,
                 distance_threshold: float = 0.15,
                 peak_prominence: float = 10.0,
                 min_distance_between_touches: int = 5,
                 merge_window: int = 3):
        """
        TouchDetector 초기화

        Args:
            velocity_threshold: 속도 최소값 임계값 (pixels/frame)
            distance_threshold: 발-공 거리 임계값 (미터)
            peak_prominence: 속도 피크 감지 민감도
            min_distance_between_touches: 터치 간 최소 프레임 거리
            merge_window: 두 방법 결과 병합 윈도우 (프레임)
        """
        self.velocity_threshold = velocity_threshold
        self.distance_threshold = distance_threshold
        self.peak_prominence = peak_prominence
        self.min_distance_between_touches = min_distance_between_touches
        self.merge_window = merge_window

    def detect_touches(self, pose_frames: List, ball_motion_data=None) -> List[TouchEvent]:
        """
        포즈 프레임과 공 데이터에서 터치 순간 감지

        Args:
            pose_frames: PoseFrame3D 또는 PoseFrame 리스트
            ball_motion_data: BallMotionData 객체 (선택)

        Returns:
            List[TouchEvent]: 감지된 터치 이벤트 리스트
        """
        if not pose_frames:
            return []

        # 공 위치 및 속도 추출
        ball_positions = self._extract_ball_positions(pose_frames)
        ball_velocities = self._calculate_velocities(ball_positions, pose_frames)

        # 방법 1: 속도 기반 감지
        velocity_touches = self._find_velocity_minima(ball_velocities, pose_frames)

        # 방법 2: 거리 기반 감지
        distance_touches = self._find_close_to_foot(pose_frames, ball_positions)

        # 두 방법 교차 검증 및 병합
        merged_touches = self._merge_detections(velocity_touches, distance_touches, pose_frames)

        return merged_touches

    def _extract_ball_positions(self, pose_frames: List) -> np.ndarray:
        """포즈 프레임에서 공 위치 추출"""
        positions = []

        for pf in pose_frames:
            if pf.ball_position is not None:
                positions.append([pf.frame_number, pf.ball_position[0], pf.ball_position[1]])
            else:
                # 공이 감지되지 않은 프레임은 NaN
                positions.append([pf.frame_number, np.nan, np.nan])

        return np.array(positions)

    def _calculate_velocities(self, ball_positions: np.ndarray, pose_frames: List) -> np.ndarray:
        """공 위치에서 속도 계산"""
        velocities = np.zeros(len(ball_positions))

        for i in range(1, len(ball_positions)):
            if np.isnan(ball_positions[i, 1]) or np.isnan(ball_positions[i-1, 1]):
                velocities[i] = np.nan
                continue

            dx = ball_positions[i, 1] - ball_positions[i-1, 1]
            dy = ball_positions[i, 2] - ball_positions[i-1, 2]
            velocities[i] = np.sqrt(dx**2 + dy**2)

        return velocities

    def _find_velocity_minima(self, velocities: np.ndarray, pose_frames: List) -> List[TouchEvent]:
        """
        속도 최소값에서 터치 감지

        공이 발에 닿는 순간 속도가 급격히 감소했다가
        다시 증가하는 패턴을 찾습니다.
        """
        touches = []

        # NaN 처리를 위한 보간
        valid_mask = ~np.isnan(velocities)
        if np.sum(valid_mask) < 10:
            return touches

        # 속도를 반전시켜 최소값을 피크로 감지
        inverted = np.nanmax(velocities) - velocities

        # 피크 감지
        peaks, properties = find_peaks(
            np.nan_to_num(inverted, nan=0),
            prominence=self.peak_prominence,
            distance=self.min_distance_between_touches
        )

        for peak in peaks:
            if peak >= len(pose_frames):
                continue

            # 속도가 임계값 이하인지 확인
            if not np.isnan(velocities[peak]) and velocities[peak] < self.velocity_threshold * 3:
                pf = pose_frames[peak]
                touch = TouchEvent(
                    frame_number=pf.frame_number,
                    timestamp=pf.timestamp,
                    touch_type='velocity',
                    confidence=0.7,
                    ball_position=pf.ball_position
                )
                touches.append(touch)

        return touches

    def _find_close_to_foot(self, pose_frames: List, ball_positions: np.ndarray) -> List[TouchEvent]:
        """
        발-공 거리가 가까운 순간에서 터치 감지

        양 발목의 위치와 공의 위치를 비교하여
        거리가 임계값 이하인 순간을 찾습니다.
        """
        touches = []

        for i, pf in enumerate(pose_frames):
            if pf.ball_position is None:
                continue

            # 발 위치 추출 (3D 또는 2D)
            left_foot, right_foot = self._get_foot_positions(pf)
            if left_foot is None or right_foot is None:
                continue

            # 공 위치 (2D → 발 위치와 비교하려면 정규화 필요)
            ball_x = pf.ball_position[0] / pf.frame_width if pf.frame_width > 0 else 0
            ball_y = pf.ball_position[1] / pf.frame_height if pf.frame_height > 0 else 0

            # 발-공 거리 계산 (2D 정규화 좌표)
            left_dist = np.sqrt((left_foot[0] - ball_x)**2 + (left_foot[1] - ball_y)**2)
            right_dist = np.sqrt((right_foot[0] - ball_x)**2 + (right_foot[1] - ball_y)**2)

            min_dist = min(left_dist, right_dist)
            foot_side = 'left' if left_dist < right_dist else 'right'

            # 거리 임계값 이하이면 터치
            if min_dist < self.distance_threshold:
                touch = TouchEvent(
                    frame_number=pf.frame_number,
                    timestamp=pf.timestamp,
                    touch_type='distance',
                    foot_side=foot_side,
                    confidence=1.0 - min_dist / self.distance_threshold,
                    ball_position=pf.ball_position,
                    foot_position=left_foot if foot_side == 'left' else right_foot
                )
                touches.append(touch)

        # 연속된 프레임 중 최소 거리만 유지
        return self._filter_consecutive(touches)

    def _get_foot_positions(self, pose_frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """포즈 프레임에서 양 발 위치 추출"""
        # PoseFrame3D (SMPL-X)
        if hasattr(pose_frame, 'joints_3d'):
            # SMPL-X: left_ankle=7, right_ankle=8
            left_foot = pose_frame.joints_2d[7] if pose_frame.joints_2d is not None else None
            right_foot = pose_frame.joints_2d[8] if pose_frame.joints_2d is not None else None
            return left_foot, right_foot

        # PoseFrame (MediaPipe)
        if hasattr(pose_frame, 'landmarks'):
            # MediaPipe: LEFT_ANKLE=27, RIGHT_ANKLE=28
            landmarks = pose_frame.landmarks
            if landmarks is not None and len(landmarks) > 28:
                left_foot = landmarks[27][:2]  # x, y만 사용
                right_foot = landmarks[28][:2]
                return left_foot, right_foot

        return None, None

    def _filter_consecutive(self, touches: List[TouchEvent]) -> List[TouchEvent]:
        """연속된 터치 이벤트 중 가장 신뢰도 높은 것만 유지"""
        if not touches:
            return touches

        filtered = []
        current_group = [touches[0]]

        for i in range(1, len(touches)):
            if touches[i].frame_number - touches[i-1].frame_number <= self.min_distance_between_touches:
                current_group.append(touches[i])
            else:
                # 그룹에서 가장 높은 신뢰도 선택
                best = max(current_group, key=lambda t: t.confidence)
                filtered.append(best)
                current_group = [touches[i]]

        # 마지막 그룹 처리
        if current_group:
            best = max(current_group, key=lambda t: t.confidence)
            filtered.append(best)

        return filtered

    def _merge_detections(self, velocity_touches: List[TouchEvent],
                          distance_touches: List[TouchEvent],
                          pose_frames: List) -> List[TouchEvent]:
        """
        두 감지 방법의 결과 병합

        같은 시간대에 두 방법 모두 감지하면 신뢰도 상승
        """
        merged = []
        used_velocity = set()
        used_distance = set()

        # 속도와 거리 감지 결과 매칭
        for vt in velocity_touches:
            for dt in distance_touches:
                if abs(vt.frame_number - dt.frame_number) <= self.merge_window:
                    # 두 방법 모두 감지 → 높은 신뢰도
                    merged_touch = TouchEvent(
                        frame_number=(vt.frame_number + dt.frame_number) // 2,
                        timestamp=(vt.timestamp + dt.timestamp) / 2,
                        touch_type='both',
                        foot_side=dt.foot_side,
                        confidence=min(1.0, vt.confidence + dt.confidence),
                        ball_position=dt.ball_position or vt.ball_position,
                        foot_position=dt.foot_position
                    )
                    merged.append(merged_touch)
                    used_velocity.add(vt.frame_number)
                    used_distance.add(dt.frame_number)
                    break

        # 매칭되지 않은 속도 감지 추가
        for vt in velocity_touches:
            if vt.frame_number not in used_velocity:
                merged.append(vt)

        # 매칭되지 않은 거리 감지 추가 (신뢰도 높은 것만)
        for dt in distance_touches:
            if dt.frame_number not in used_distance and dt.confidence > 0.5:
                merged.append(dt)

        # 프레임 번호로 정렬
        merged.sort(key=lambda t: t.frame_number)

        # 최종 필터링
        return self._filter_consecutive(merged)

    def get_touch_frames(self, touches: List[TouchEvent]) -> List[int]:
        """터치 이벤트 리스트에서 프레임 번호만 추출"""
        return [t.frame_number for t in touches]

    def get_touch_phases(self, pose_frames: List, touches: List[TouchEvent],
                         window: int = 5) -> List[Tuple[int, int, str]]:
        """
        각 프레임의 터치 단계 결정

        Returns:
            List[(frame_number, touch_index, phase)]
            phase: 'before', 'touch', 'after'
        """
        phases = []

        for pf in pose_frames:
            frame = pf.frame_number
            phase = 'none'
            touch_idx = -1

            for i, touch in enumerate(touches):
                if abs(frame - touch.frame_number) <= window:
                    touch_idx = i
                    if frame < touch.frame_number:
                        phase = 'before'
                    elif frame == touch.frame_number:
                        phase = 'touch'
                    else:
                        phase = 'after'
                    break

            phases.append((frame, touch_idx, phase))

        return phases
