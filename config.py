# config.py
"""
Soccer Motion Analysis - Configuration
"""

# MediaPipe 설정
MEDIAPIPE_CONFIG = {
    'model_complexity': 2,  # 0, 1, 2 (높을수록 정확하지만 느림)
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'enable_segmentation': False,
    'smooth_landmarks': True
}

# 분석 설정
ANALYSIS_CONFIG = {
    'smooth_window': 5,  # 각도 smoothing window size
    'confidence_threshold': 0.6,  # 최소 신뢰도
}

# 관절 및 분절 설정
JOINTS_TO_ANALYZE = [
    'left_knee',
    'right_knee',
    'left_hip',
    'right_hip',
    'left_ankle',
    'right_ankle'
]

SEGMENTS_TO_ANALYZE = [
    'trunk',
    'left_thigh',
    'right_thigh',
    'left_shank',
    'right_shank',
    'left_foot',
    'right_foot'
]

# MediaPipe Landmark 인덱스
LANDMARK_INDICES = {
    'NOSE': 0,
    'LEFT_EYE': 2,
    'RIGHT_EYE': 5,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_ELBOW': 13,
    'RIGHT_ELBOW': 14,
    'LEFT_WRIST': 15,
    'RIGHT_WRIST': 16,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_HEEL': 29,
    'RIGHT_HEEL': 30,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32
}

# 시각화 설정
VISUALIZATION_CONFIG = {
    'skeleton_color': (0, 255, 0),  # 초록색
    'video1_color': (255, 0, 0),    # 파란색 (BGR)
    'video2_color': (0, 0, 255),    # 빨간색 (BGR)
    'line_thickness': 2,
    'point_radius': 4,
    'figure_size': (12, 8),
    'dpi': 100
}

# 출력 설정
OUTPUT_CONFIG = {
    'save_data': True,
    'save_video': True,
    'save_plots': True,
    'data_format': 'csv',  # 'csv', 'json'
    'video_codec': 'mp4v',
    'video_fps': 30
}

# 해석 임계값
INTERPRETATION_THRESHOLDS = {
    'angle_difference': {
        'very_similar': 3,    # < 3도: 매우 유사
        'similar': 5,         # < 5도: 유사
        'moderate': 10,       # < 10도: 약간 다름
        # >= 10도: 많이 다름
    },
    'rom_difference': {
        'very_similar': 3,
        'similar': 5,
        'moderate': 10,
    }
}

# 드리블 스타일 분류 기준
STYLE_CLASSIFICATION = {
    'knee_angle': {
        'very_low': 135,   # < 135도: 매우 낮은 자세
        'low': 145,        # < 145도: 낮은 자세
        'medium': 155,     # < 155도: 중간 자세
        # >= 155도: 높은 자세
    },
    'trunk_lean': {
        'aggressive': 15,  # > 15도 숙임: 공격적
        'moderate': 10,    # > 10도 숙임: 보통
        # < 10도 숙임: 안정적
    }
}