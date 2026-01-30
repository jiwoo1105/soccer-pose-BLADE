# analysis/ball_motion_analyzer.py
"""공의 움직임 분석 및 터치 순간 감지 모듈"""

import numpy as np
from typing import List, Optional
from scipy.signal import find_peaks, savgol_filter
from dataclasses import dataclass


@dataclass
class BallMotionData:
    """공의 움직임 분석 결과"""
    frame_numbers: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    velocity_magnitudes: np.ndarray
    touch_frames: List[int]
    touch_count: int

    def __str__(self):
        result = []
        result.append("="*70)
        result.append("공 움직임 분석 결과")
        result.append("="*70)
        result.append(f"총 프레임 수: {len(self.frame_numbers)}")
        result.append(f"터치 횟수: {self.touch_count}")
        result.append(f"터치 프레임: {self.touch_frames}")
        result.append(f"평균 속도: {np.mean(self.velocity_magnitudes):.2f} pixels/frame")
        result.append(f"최대 속도: {np.max(self.velocity_magnitudes):.2f} pixels/frame")
        result.append("="*70)
        return "\n".join(result)


class BallMotionAnalyzer:
    """공의 움직임 분석 및 터치 순간 감지"""

    def __init__(self,
                 min_velocity_threshold: float = 5.0,
                 peak_prominence: float = 10.0,
                 min_distance_between_touches: int = 12,
                 use_smoothing: bool = True,
                 smoothing_window: int = 5):
        self.min_velocity_threshold = min_velocity_threshold
        self.peak_prominence = peak_prominence
        self.min_distance_between_touches = min_distance_between_touches
        self.use_smoothing = use_smoothing
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1

    def _smooth_positions(self, positions: np.ndarray) -> np.ndarray:
        """Savitzky-Golay 필터로 위치 데이터 스무딩"""
        if len(positions) < self.smoothing_window:
            return positions

        smoothed = positions.copy()
        polyorder = min(3, self.smoothing_window - 2)
        smoothed[:, 0] = savgol_filter(positions[:, 0], self.smoothing_window, polyorder)
        smoothed[:, 1] = savgol_filter(positions[:, 1], self.smoothing_window, polyorder)
        return smoothed

    def analyze(self, pose_frames) -> Optional[BallMotionData]:
        """PoseFrame 리스트에서 공의 움직임 분석"""
        ball_frames = [(pf.frame_number, pf.ball_position)
                      for pf in pose_frames
                      if pf.ball_position is not None]

        if len(ball_frames) == 0:
            return None

        frame_numbers = np.array([f[0] for f in ball_frames])
        positions = np.array([f[1] for f in ball_frames])

        if self.use_smoothing:
            positions = self._smooth_positions(positions)

        velocities = self._calculate_velocities(positions, frame_numbers)
        velocity_magnitudes = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
        touch_indices = self._detect_peaks(velocity_magnitudes)
        touch_frames = [int(frame_numbers[idx]) for idx in touch_indices]

        return BallMotionData(
            frame_numbers=frame_numbers,
            positions=positions,
            velocities=velocities,
            velocity_magnitudes=velocity_magnitudes,
            touch_frames=touch_frames,
            touch_count=len(touch_frames)
        )

    def _calculate_velocities(self, positions: np.ndarray,
                             frame_numbers: np.ndarray) -> np.ndarray:
        """위치 데이터에서 속도 계산"""
        N = len(positions)
        velocities = np.zeros((N, 2))

        for i in range(1, N):
            dt = frame_numbers[i] - frame_numbers[i-1]
            if dt > 0:
                velocities[i] = (positions[i] - positions[i-1]) / dt

        velocities[0] = velocities[1] if N > 1 else [0, 0]
        return velocities

    def _detect_peaks(self, velocity_magnitudes: np.ndarray) -> List[int]:
        """속도 극소값 감지로 터치 순간 찾기"""
        inverted = -velocity_magnitudes
        peaks, _ = find_peaks(
            inverted,
            prominence=self.peak_prominence,
            distance=self.min_distance_between_touches
        )

        valid_peaks = [
            p for p in peaks
            if velocity_magnitudes[p] < self.min_velocity_threshold
        ]
        return valid_peaks
