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


def project_to_horizontal(v: np.ndarray) -> np.ndarray:
    """
    벡터를 수평면(X-Z)에 투영

    3D 벡터에서 Y 성분을 제거하여 수평면에 투영합니다.
    정렬(alignment) 계산에 사용됩니다.

    Args:
        v: 3D 벡터

    Returns:
        np.ndarray: 수평면에 투영된 정규화 벡터
    """
    if len(v) < 3:
        return v

    horizontal = np.array([v[0], 0, v[2]])
    norm = np.linalg.norm(horizontal)

    if norm < 1e-10:
        return np.array([1, 0, 0])

    return horizontal / norm


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """벡터 정규화"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """3D 벡터 외적"""
    return np.cross(v1, v2)


def calculate_gaze_angle(eye_center: np.ndarray, target: np.ndarray) -> float:
    """
    시선 각도 계산 (수평 기준)

    Args:
        eye_center: 눈 중심 위치
        target: 시선 목표 위치 (턱 또는 코)

    Returns:
        float: 시선 각도 (도, 아래가 양수)
    """
    gaze_vector = target - eye_center

    if len(gaze_vector) < 3:
        return 0.0

    horizontal_length = np.sqrt(gaze_vector[0]**2 + gaze_vector[2]**2)
    vertical = abs(gaze_vector[1])

    if horizontal_length < 1e-10:
        return 90.0 if gaze_vector[1] > 0 else -90.0

    angle_rad = np.arctan2(vertical, horizontal_length)
    angle_deg = np.degrees(angle_rad)

    return angle_deg if gaze_vector[1] > 0 else -angle_deg


def calculate_body_forward(left_shoulder: np.ndarray, right_shoulder: np.ndarray,
                           vertical: np.ndarray = None) -> np.ndarray:
    """
    몸의 전방 방향 계산

    어깨 벡터와 수직 벡터의 외적으로 전방 방향을 계산합니다.

    Args:
        left_shoulder: 왼쪽 어깨 위치
        right_shoulder: 오른쪽 어깨 위치
        vertical: 수직 방향 벡터 (기본값: [0, 1, 0])

    Returns:
        np.ndarray: 전방 방향 단위 벡터
    """
    if vertical is None:
        vertical = np.array([0, 1, 0])

    shoulder_vec = right_shoulder - left_shoulder
    forward = np.cross(shoulder_vec, vertical)

    return normalize_vector(forward)
