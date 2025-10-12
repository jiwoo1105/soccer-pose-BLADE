# analysis/joint_analyzer.py
"""
관절 각도 분석
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from utils.math_utils import calculate_angle


class JointAnalyzer:
    """
    관절 각도 분석
    - Knee (무릎)
    - Hip (고관절)
    - Ankle (발목)
    """
    
    def calculate_knee_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        무릎 각도 계산 (hip - knee - ankle)
        
        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'
            
        Returns:
            float: 무릎 각도 (degrees) - 180도가 완전히 펴진 상태
        """
        if side == 'left':
            hip = landmarks[23]
            knee = landmarks[25]
            ankle = landmarks[27]
        else:
            hip = landmarks[24]
            knee = landmarks[26]
            ankle = landmarks[28]
        
        angle = calculate_angle(hip, knee, ankle)
        
        return angle
    
    def calculate_hip_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        고관절 각도 계산 (shoulder - hip - knee)
        
        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'
            
        Returns:
            float: 고관절 각도 (degrees)
        """
        if side == 'left':
            shoulder = landmarks[11]
            hip = landmarks[23]
            knee = landmarks[25]
        else:
            shoulder = landmarks[12]
            hip = landmarks[24]
            knee = landmarks[26]
        
        angle = calculate_angle(shoulder, hip, knee)
        
        return angle
    
    def calculate_ankle_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        발목 각도 계산 (knee - ankle - foot)
        
        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'
            
        Returns:
            float: 발목 각도 (degrees)
        """
        if side == 'left':
            knee = landmarks[25]
            ankle = landmarks[27]
            foot = landmarks[31]  # foot index
        else:
            knee = landmarks[26]
            ankle = landmarks[28]
            foot = landmarks[32]
        
        angle = calculate_angle(knee, ankle, foot)
        
        return angle
    
    def calculate_all_joints(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        모든 관절 각도 계산
        
        Args:
            landmarks: (33, 3) landmarks array
            
        Returns:
            Dict[str, float]: 각 관절의 각도
        """
        joints = {
            'left_knee': self.calculate_knee_angle(landmarks, 'left'),
            'right_knee': self.calculate_knee_angle(landmarks, 'right'),
            'left_hip': self.calculate_hip_angle(landmarks, 'left'),
            'right_hip': self.calculate_hip_angle(landmarks, 'right'),
            'left_ankle': self.calculate_ankle_angle(landmarks, 'left'),
            'right_ankle': self.calculate_ankle_angle(landmarks, 'right')
        }
        
        return joints