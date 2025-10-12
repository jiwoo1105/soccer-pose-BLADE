# core/video_processor.py
"""
비디오 읽기/쓰기 및 기본 처리
"""

import cv2
import numpy as np
from typing import Generator, Dict, Optional


class VideoProcessor:
    """
    비디오 파일 처리 유틸리티
    """
    
    def __init__(self, video_path: str):
        """
        Args:
            video_path: 비디오 파일 경로
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
    
    def get_video_info(self) -> Dict:
        """
        비디오 정보 반환
        
        Returns:
            Dict: 비디오 메타데이터
        """
        return {
            'path': self.video_path,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'total_frames': self.total_frames,
            'duration': self.duration
        }
    
    def read_frames(self) -> Generator[np.ndarray, None, None]:
        """
        비디오의 모든 프레임을 순회
        
        Yields:
            np.ndarray: BGR 이미지 프레임
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 처음부터
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame
    
    def read_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        특정 프레임 읽기
        
        Args:
            frame_number: 프레임 번호 (0부터 시작)
            
        Returns:
            np.ndarray or None: BGR 이미지
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    @staticmethod
    def save_video(frames: list, 
                   output_path: str, 
                   fps: float = 30.0,
                   codec: str = 'mp4v') -> None:
        """
        프레임들을 비디오로 저장
        
        Args:
            frames: 프레임 리스트
            output_path: 출력 파일 경로
            fps: 프레임 레이트
            codec: 비디오 코덱
        """
        if not frames:
            print("No frames to save")
            return
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()
        print(f"Video saved to {output_path}")
    
    def resize_frame(self, frame: np.ndarray, 
                    target_width: int = None, 
                    target_height: int = None) -> np.ndarray:
        """
        프레임 크기 조정
        
        Args:
            frame: 입력 프레임
            target_width: 목표 너비
            target_height: 목표 높이
            
        Returns:
            np.ndarray: 크기 조정된 프레임
        """
        if target_width and target_height:
            return cv2.resize(frame, (target_width, target_height))
        elif target_width:
            aspect_ratio = frame.shape[0] / frame.shape[1]
            target_height = int(target_width * aspect_ratio)
            return cv2.resize(frame, (target_width, target_height))
        elif target_height:
            aspect_ratio = frame.shape[1] / frame.shape[0]
            target_width = int(target_height * aspect_ratio)
            return cv2.resize(frame, (target_width, target_height))
        else:
            return frame
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'cap'):
            self.cap.release()