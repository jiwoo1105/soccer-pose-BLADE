# core/__init__.py
"""
Core modules for pose extraction and video processing
"""

from .pose_extractor import PoseExtractor
from .video_processor import VideoProcessor
from .coordinate_system import CoordinateSystem

__all__ = ['PoseExtractor', 'VideoProcessor', 'CoordinateSystem']