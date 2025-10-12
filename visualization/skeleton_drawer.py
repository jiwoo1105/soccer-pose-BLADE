# visualization/skeleton_drawer.py
"""
스켈레톤 시각화
"""

import cv2
import numpy as np
from typing import Tuple


class SkeletonDrawer:
    """
    MediaPipe 포즈를 영상에 그리기
    """
    
    # MediaPipe 연결선 정의
    CONNECTIONS = [
        # 몸통
        (11, 12),  # 어깨
        (11, 23),  # 왼쪽 어깨-엉덩이
        (12, 24),  # 오른쪽 어깨-엉덩이
        (23, 24),  # 엉덩이
        
        # 왼쪽 팔
        (11, 13),  # 어깨-팔꿈치
        (13, 15),  # 팔꿈치-손목
        
        # 오른쪽 팔
        (12, 14),
        (14, 16),
        
        # 왼쪽 다리
        (23, 25),  # 엉덩이-무릎
        (25, 27),  # 무릎-발목
        (27, 29),  # 발목-발뒤꿈치
        (29, 31),  # 발뒤꿈치-발끝
        
        # 오른쪽 다리
        (24, 26),
        (26, 28),
        (28, 30),
        (30, 32),
    ]
    
    def __init__(self, color: Tuple[int, int, int] = (0, 255, 0), 
                 thickness: int = 2,
                 point_radius: int = 4):
        """
        Args:
            color: 선 색상 (BGR)
            thickness: 선 두께
            point_radius: 점 반지름
        """
        self.color = color
        self.thickness = thickness
        self.point_radius = point_radius
    
    def draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        프레임에 스켈레톤 그리기
        
        Args:
            frame: BGR 이미지
            landmarks: (33, 3) normalized landmarks [0-1 range]
            
        Returns:
            np.ndarray: 스켈레톤이 그려진 이미지
        """
        h, w, _ = frame.shape
        output = frame.copy()
        
        # 연결선 그리기
        for start_idx, end_idx in self.CONNECTIONS:
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            
            # Normalized coordinates를 픽셀로 변환
            start_pixel = (int(start_point[0] * w), int(start_point[1] * h))
            end_pixel = (int(end_point[0] * w), int(end_point[1] * h))
            
            cv2.line(output, start_pixel, end_pixel, self.color, self.thickness)
        
        # 관절점 그리기
        for landmark in landmarks:
            pixel = (int(landmark[0] * w), int(landmark[1] * h))
            cv2.circle(output, pixel, self.point_radius, self.color, -1)
        
        return output
    
    def draw_angle_text(self, frame: np.ndarray, 
                       text: str, 
                       position: Tuple[int, int],
                       color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        프레임에 텍스트 그리기
        
        Args:
            frame: BGR 이미지
            text: 표시할 텍스트
            position: (x, y) 위치
            color: 텍스트 색상
            
        Returns:
            np.ndarray: 텍스트가 추가된 이미지
        """
        output = frame.copy()
        
        # 배경 박스
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        cv2.rectangle(output, 
                     (position[0] - 5, position[1] - text_h - 5),
                     (position[0] + text_w + 5, position[1] + 5),
                     (0, 0, 0), -1)
        
        # 텍스트
        cv2.putText(output, text, position, font, font_scale, color, thickness)
        
        return output