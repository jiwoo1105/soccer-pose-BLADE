# core/__init__.py
"""
Core modules for soccer motion analysis

주요 모듈:
- PoseExtractor: BLADE 기반 3D 포즈 추출
- BladeWrapper: BLADE API 래퍼 (SMPL-X 54개 관절)
- HybridBallDetector: YOLOv9 + 스켈레톤 폴백 + Kalman 필터
- TouchDetector: 하이브리드 터치 감지
"""

from .pose_extractor import PoseExtractor
from .blade_wrapper import BladeWrapper, PoseFrame3D, SMPLX_JOINT_INDICES
try:
    from .ball_detector import HybridBallDetector, BallDetector
    _BALL_DETECTOR_AVAILABLE = True
except Exception:
    HybridBallDetector = None
    BallDetector = None
    _BALL_DETECTOR_AVAILABLE = False
from .touch_detector import TouchDetector, TouchEvent

__all__ = [
    # 포즈 추출
    'PoseExtractor',
    'BladeWrapper',
    'PoseFrame3D',
    'SMPLX_JOINT_INDICES',
    # 공 탐지
    'HybridBallDetector',
    'BallDetector',
    # 터치 감지
    'TouchDetector',
    'TouchEvent',
]
