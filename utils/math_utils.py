# utils/math_utils.py
"""
수학 유틸리티 함수들
"""

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    두 벡터 사이의 각도 계산
    
    Args:
        v1: 벡터 1
        v2: 벡터 2
        
    Returns:
        float: 각도 (degrees)
    """
    v1_normalized = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_normalized = v2 / (np.linalg.norm(v2) + 1e-10)
    
    cos_angle = np.dot(v1_normalized, v2_normalized)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def angle_with_vertical(vector: np.ndarray) -> float:
    """
    벡터와 수직축(Y축)의 각도 계산
    
    Args:
        vector: 3D 벡터
        
    Returns:
        float: 각도 (degrees)
    """
    vertical = np.array([0, -1, 0])  # 아래 방향 (MediaPipe 좌표계)
    return angle_between_vectors(vector, vertical)


def angle_with_horizontal(vector: np.ndarray) -> float:
    """
    벡터와 수평축(X-Z 평면)의 각도 계산
    
    Args:
        vector: 3D 벡터
        
    Returns:
        float: 각도 (degrees)
    """
    # X-Z 평면으로 투영
    horizontal_vector = np.array([vector[0], 0, vector[2]])
    
    if np.linalg.norm(horizontal_vector) < 1e-10:
        return 90.0  # 완전히 수직
    
    return angle_between_vectors(vector, horizontal_vector)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    세 점으로 정의된 각도 계산 (p1-p2-p3, p2가 꼭지점)
    
    Args:
        p1: 첫 번째 점
        p2: 중간 점 (꼭지점)
        p3: 세 번째 점
        
    Returns:
        float: 각도 (degrees)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    
    return angle_between_vectors(v1, v2)


def smooth_angles(angles: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    각도 시계열 데이터를 smoothing
    
    Args:
        angles: 각도 배열
        window_size: 윈도우 크기 (홀수 권장)
        
    Returns:
        np.ndarray: smoothed 각도 배열
    """
    if len(angles) < window_size:
        return angles
    
    # Moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(angles, kernel, mode='same')
    
    return smoothed