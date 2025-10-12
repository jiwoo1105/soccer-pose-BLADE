# core/pose_extractor.py
"""
MediaPipe를 사용한 3D 포즈 추출
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class PoseFrame:
    """단일 프레임의 포즈 데이터"""
    frame_number: int
    timestamp: float
    landmarks: np.ndarray  # (33, 3) - normalized coordinates
    world_landmarks: np.ndarray  # (33, 3) - real world coordinates in meters
    visibility: np.ndarray  # (33,) - visibility scores


class PoseExtractor:
    """
    MediaPipe Pose를 사용한 3D 포즈 추출
    """
    
    def __init__(self, 
                 model_complexity: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            model_complexity: 0, 1, 2 (높을수록 정확하지만 느림)
            min_detection_confidence: 최소 감지 신뢰도
            min_tracking_confidence: 최소 추적 신뢰도
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            enable_segmentation=False,
            smooth_landmarks=True
        )
        
    def extract_from_video(self, video_path: str) -> List[PoseFrame]:
        """
        비디오에서 모든 프레임의 포즈 추출
        
        Args:
            video_path: 비디오 파일 경로
            
        Returns:
            List[PoseFrame]: 각 프레임의 포즈 데이터 리스트
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_frames = []
        frame_number = 0
        
        print(f"Extracting poses from {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # RGB로 변환 (MediaPipe 요구사항)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 포즈 추출
                results = self.pose.process(frame_rgb)
                
                if results.pose_landmarks and results.pose_world_landmarks:
                    timestamp = frame_number / fps
                    
                    # Landmarks 추출
                    landmarks = self._extract_landmarks(results.pose_landmarks)
                    world_landmarks = self._extract_landmarks(results.pose_world_landmarks)
                    visibility = self._extract_visibility(results.pose_landmarks)
                    
                    pose_frame = PoseFrame(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        landmarks=landmarks,
                        world_landmarks=world_landmarks,
                        visibility=visibility
                    )
                    
                    pose_frames.append(pose_frame)
                
                frame_number += 1
                pbar.update(1)
        
        cap.release()
        
        print(f"Successfully extracted {len(pose_frames)} frames with poses")
        
        return pose_frames
    
    def _extract_landmarks(self, pose_landmarks) -> np.ndarray:
        """
        Landmarks를 numpy array로 변환
        
        Returns:
            np.ndarray: (33, 3) array of (x, y, z) coordinates
        """
        landmarks = np.array([
            [lm.x, lm.y, lm.z] 
            for lm in pose_landmarks.landmark
        ])
        return landmarks
    
    def _extract_visibility(self, pose_landmarks) -> np.ndarray:
        """
        Visibility 점수 추출
        
        Returns:
            np.ndarray: (33,) array of visibility scores
        """
        visibility = np.array([
            lm.visibility 
            for lm in pose_landmarks.landmark
        ])
        return visibility
    
    def extract_from_frame(self, frame: np.ndarray) -> Optional[PoseFrame]:
        """
        단일 프레임에서 포즈 추출
        
        Args:
            frame: BGR 이미지
            
        Returns:
            PoseFrame or None
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks and results.pose_world_landmarks:
            landmarks = self._extract_landmarks(results.pose_landmarks)
            world_landmarks = self._extract_landmarks(results.pose_world_landmarks)
            visibility = self._extract_visibility(results.pose_landmarks)
            
            return PoseFrame(
                frame_number=0,
                timestamp=0.0,
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                visibility=visibility
            )
        
        return None
    
    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'pose'):
            self.pose.close()