# core/coordinate_system.py
"""
좌표계 변환 및 정규화
"""

import numpy as np
from typing import Tuple


class CoordinateSystem:
    """
    좌표계 변환 및 정규화 유틸리티
    """
    
    @staticmethod
    def normalize_by_hip_center(landmarks: np.ndarray, 
                                left_hip_idx: int = 23, 
                                right_hip_idx: int = 24) -> np.ndarray:
        """
        엉덩이 중심을 원점으로 이동
        
        Args:
            landmarks: (33, 3) landmarks array
            left_hip_idx: 왼쪽 엉덩이 인덱스
            right_hip_idx: 오른쪽 엉덩이 인덱스
            
        Returns:
            np.ndarray: 정규화된 landmarks
        """
        hip_center = (landmarks[left_hip_idx] + landmarks[right_hip_idx]) / 2
        normalized = landmarks - hip_center
        return normalized
    
    @staticmethod
    def scale_by_body_height(landmarks: np.ndarray,
                           nose_idx: int = 0,
                           left_ankle_idx: int = 27,
                           right_ankle_idx: int = 28) -> np.ndarray:
        """
        신체 높이로 스케일 정규화
        
        Args:
            landmarks: (33, 3) landmarks array
            nose_idx: 코 인덱스
            left_ankle_idx: 왼쪽 발목 인덱스
            right_ankle_idx: 오른쪽 발목 인덱스
            
        Returns:
            np.ndarray: 스케일 정규화된 landmarks
        """
        nose = landmarks[nose_idx]
        ankle_center = (landmarks[left_ankle_idx] + landmarks[right_ankle_idx]) / 2
        
        body_height = np.linalg.norm(nose - ankle_center)
        
        if body_height > 0:
            scaled = landmarks / body_height
        else:
            scaled = landmarks
        
        return scaled
    
    @staticmethod
    def calculate_body_center(landmarks: np.ndarray,
                            left_hip_idx: int = 23,
                            right_hip_idx: int = 24) -> np.ndarray:
        """
        신체 중심점 계산
        
        Args:
            landmarks: (33, 3) landmarks array
            
        Returns:
            np.ndarray: (3,) 중심점 좌표
        """
        hip_center = (landmarks[left_hip_idx] + landmarks[right_hip_idx]) / 2
        return hip_center
    
    @staticmethod
    def estimate_ground_plane(landmarks: np.ndarray,
                            left_heel_idx: int = 29,
                            right_heel_idx: int = 30,
                            left_foot_idx: int = 31,
                            right_foot_idx: int = 32) -> Tuple[np.ndarray, float]:
        """
        지면 평면 추정
        
        Args:
            landmarks: (33, 3) landmarks array
            
        Returns:
            Tuple[np.ndarray, float]: (법선 벡터, d 상수)
        """
        # 발 랜드마크로 평면 정의
        foot_points = np.array([
            landmarks[left_heel_idx],
            landmarks[right_heel_idx],
            landmarks[left_foot_idx],
            landmarks[right_foot_idx]
        ])
        
        # 평면의 중심
        centroid = np.mean(foot_points, axis=0)
        
        # SVD로 평면의 법선 벡터 찾기
        centered = foot_points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # 가장 작은 특이값에 대응하는 벡터
        
        # 법선 벡터가 위쪽을 가리키도록 조정
        if normal[1] < 0:
            normal = -normal
        
        # 평면 방정식: normal · (x - centroid) = 0
        d = -np.dot(normal, centroid)
        
        return normal, d
    
    @staticmethod
    def rotate_to_align_with_axis(landmarks: np.ndarray, 
                                  target_axis: str = 'y') -> np.ndarray:
        """
        특정 축과 정렬하도록 회전
        
        Args:
            landmarks: (33, 3) landmarks array
            target_axis: 'x', 'y', 또는 'z'
            
        Returns:
            np.ndarray: 회전된 landmarks
        """
        # 몸통 벡터 계산 (어깨 중심 - 엉덩이 중심)
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        trunk_vector = shoulder_center - hip_center
        trunk_vector = trunk_vector / np.linalg.norm(trunk_vector)
        
        # 목표 축 벡터
        if target_axis == 'y':
            target = np.array([0, 1, 0])
        elif target_axis == 'x':
            target = np.array([1, 0, 0])
        else:  # 'z'
            target = np.array([0, 0, 1])
        
        # 회전 축과 각도 계산
        rotation_axis = np.cross(trunk_vector, target)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            return landmarks  # 이미 정렬됨
        
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(trunk_vector, target), -1.0, 1.0))
        
        # Rodrigues' rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # 모든 landmarks 회전
        rotated = landmarks @ R.T
        
        return rotated