# analysis/segment_analyzer.py
"""
Body Segment 각도 분석
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from utils.math_utils import angle_between_vectors, angle_with_vertical, angle_with_horizontal


class SegmentAnalyzer:
    """
    신체 분절(Body Segment) 각도 분석
    - Trunk (몸통)
    - Thigh (허벅지)
    - Shank (정강이)
    - Foot (발)
    """
    
    def calculate_trunk_angle(self, landmarks: np.ndarray) -> float:
        """
        몸통 각도 계산 (어깨 중심 - 엉덩이 중심과 수직축의 각도)
        
        Args:
            landmarks: (33, 3) landmarks array
            
        Returns:
            float: 몸통 각도 (degrees)
        """
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        trunk_vector = shoulder_center - hip_center
        
        angle = angle_with_vertical(trunk_vector)
        
        return angle
    
    def calculate_thigh_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        허벅지 각도 계산 (엉덩이 - 무릎과 수직축의 각도)
        
        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'
            
        Returns:
            float: 허벅지 각도 (degrees)
        """
        if side == 'left':
            hip = landmarks[23]
            knee = landmarks[25]
        else:
            hip = landmarks[24]
            knee = landmarks[26]
        
        thigh_vector = knee - hip
        angle = angle_with_vertical(thigh_vector)
        
        return angle
    
    def calculate_shank_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        정강이 각도 계산 (무릎 - 발목과 수직축의 각도)
        
        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'
            
        Returns:
            float: 정강이 각도 (degrees)
        """
        if side == 'left':
            knee = landmarks[25]
            ankle = landmarks[27]
        else:
            knee = landmarks[26]
            ankle = landmarks[28]
        
        shank_vector = ankle - knee
        angle = angle_with_vertical(shank_vector)
        
        return angle
    
    def calculate_foot_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        발 각도 계산 (발뒤꿈치 - 발끝과 수평축의 각도)
        
        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'
            
        Returns:
            float: 발 각도 (degrees)
        """
        if side == 'left':
            heel = landmarks[29]
            foot_index = landmarks[31]
        else:
            heel = landmarks[30]
            foot_index = landmarks[32]
        
        foot_vector = foot_index - heel
        angle = angle_with_horizontal(foot_vector)
        
        return angle
    
    def calculate_all_segments(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        모든 분절 각도 계산
        
        Args:
            landmarks: (33, 3) landmarks array
            
        Returns:
            Dict[str, float]: 각 분절의 각도
        """
        segments = {
            'trunk': self.calculate_trunk_angle(landmarks),
            'left_thigh': self.calculate_thigh_angle(landmarks, 'left'),
            'right_thigh': self.calculate_thigh_angle(landmarks, 'right'),
            'left_shank': self.calculate_shank_angle(landmarks, 'left'),
            'right_shank': self.calculate_shank_angle(landmarks, 'right'),
            'left_foot': self.calculate_foot_angle(landmarks, 'left'),
            'right_foot': self.calculate_foot_angle(landmarks, 'right')
        }
        
        return segments