# utils/math_utils.py
"""수학 유틸리티 함수들 - 3D 각도 계산"""

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """두 벡터 사이의 각도 계산 (degrees)"""
    v1_normalized = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_normalized = v2 / (np.linalg.norm(v2) + 1e-10)
    cos_angle = np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def angle_with_vertical(vector: np.ndarray) -> float:
    """벡터와 수직축(Y축)의 각도 계산 (degrees)"""
    vertical = np.array([0, -1, 0])
    return angle_between_vectors(vector, vertical)


def angle_with_horizontal(vector: np.ndarray) -> float:
    """벡터와 수평축(X-Z 평면)의 각도 계산 (degrees)"""
    horizontal_vector = np.array([vector[0], 0, vector[2]])
    if np.linalg.norm(horizontal_vector) < 1e-10:
        return 90.0
    return angle_between_vectors(vector, horizontal_vector)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """세 점으로 정의된 각도 계산 (p1-p2-p3, p2가 꼭지점)"""
    v1 = p1 - p2
    v2 = p3 - p2
    return angle_between_vectors(v1, v2)


def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """calculate_angle의 alias"""
    return calculate_angle(p1, p2, p3)


def smooth_angles(angles: np.ndarray, window_size: int = 5) -> np.ndarray:
    """각도 시계열 데이터를 이동 평균 필터로 smoothing"""
    if len(angles) < window_size:
        return angles
    kernel = np.ones(window_size) / window_size
    return np.convolve(angles, kernel, mode='same')
