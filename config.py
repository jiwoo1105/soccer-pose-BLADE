# config.py
"""
축구 드리블 스킬 평가 시스템 설정

주요 설정:
1. BLADE_CONFIG: BLADE 3D 포즈 추출 설정
2. MEDIAPIPE_CONFIG: MediaPipe 폴백 설정
3. BALL_DETECTION_CONFIG: 하이브리드 공 탐지 설정
4. TOUCH_DETECTION_CONFIG: 터치 감지 설정
5. SCORING_CONFIG: 4가지 평가 기준 설정
"""

# =============================================================================
# BLADE 설정 (SMPL-X 54개 관절)
# =============================================================================
BLADE_CONFIG = {
    'model_path': 'blade/pretrained/epoch_2.pth',  # BLADE 모델 가중치 경로
    'smplx_path': 'blade/body_models',       # SMPL-X 모델 경로
    'device': 'cuda:0',    # 'cuda' 또는 'cpu'
    'temporal_smoothing': True,  # 시간적 스무딩
    'smoothing_window': 5,       # 스무딩 윈도우 크기
    'batch_size': 5,             # BLADE 처리 배치 크기 (스모크 테스트용)
    'workers_per_gpu': 0,        # 데이터 로더 워커 수
    'temp_output_dir': 'output/blade_results',
    'blade_repo_path': 'blade',
    'cfg_path': 'blade/blade/configs/blade_inthewild.py',
    'max_frames': 5,          # 스모크 테스트 시 예: 5
    'resize_width': 960,      # 입력 리사이즈 (None이면 원본 유지)
    'resize_height': 540,     # 입력 리사이즈 (None이면 원본 유지)
    'enable_ball_detection': False,  # 임시로 공 탐지 비활성화 (초기 부하 제거)
}

# SMPL-X 관절 인덱스 매핑
SMPLX_JOINT_INDICES = {
    'pelvis': 0,        # 기준점
    'left_hip': 1,
    'right_hip': 2,
    'spine1': 3,
    'left_knee': 4,
    'right_knee': 5,
    'spine2': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'spine3': 9,        # 상체 각도 계산용
    'left_foot': 10,
    'right_foot': 11,
    'neck': 12,
    'left_collar': 13,
    'right_collar': 14,
    'head': 15,         # 머리 위치
    'left_shoulder': 16,  # 어깨 방향
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'jaw': 22,
    'left_eye': 23,     # 시선 방향
    'right_eye': 24,
}

# =============================================================================
# MediaPipe 설정 (폴백용)
# =============================================================================
MEDIAPIPE_CONFIG = {
    'model_complexity': 2,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
}

# MediaPipe Landmark 인덱스
LANDMARK_INDICES = {
    'NOSE': 0,
    'LEFT_EYE': 2,
    'RIGHT_EYE': 5,
    'LEFT_SHOULDER': 11,
    'RIGHT_SHOULDER': 12,
    'LEFT_HIP': 23,
    'RIGHT_HIP': 24,
    'LEFT_KNEE': 25,
    'RIGHT_KNEE': 26,
    'LEFT_ANKLE': 27,
    'RIGHT_ANKLE': 28,
    'LEFT_FOOT_INDEX': 31,
    'RIGHT_FOOT_INDEX': 32,
}

# =============================================================================
# 하이브리드 공 탐지 설정
# =============================================================================
BALL_DETECTION_CONFIG = {
    'model_name': 'yolov8n.pt',  # YOLOv9 권장: 'yolov9c.pt'
    'confidence': 0.3,
    'max_distance_threshold': 150.0,
    'history_size': 5,
    'smoothing_alpha': 0.3,
    'use_kalman': True,           # Kalman 필터 사용
    'use_skeleton_fallback': True,  # 스켈레톤 기반 폴백
}

# =============================================================================
# 터치 감지 설정
# =============================================================================
TOUCH_DETECTION_CONFIG = {
    'velocity_threshold': 5.0,      # 속도 최소값 임계값 (pixels/frame)
    'distance_threshold': 0.15,     # 발-공 거리 임계값 (정규화)
    'peak_prominence': 10.0,        # 속도 피크 감지 민감도
    'min_distance_between_touches': 5,  # 터치 간 최소 프레임 거리
    'merge_window': 3,              # 두 방법 결과 병합 윈도우
}

# 공 움직임 분석 설정 (레거시 호환)
BALL_MOTION_CONFIG = {
    'min_velocity_threshold': 5.0,
    'peak_prominence': 10.0,
    'min_distance_between_touches': 5,
    'use_smoothing': True,
    'smoothing_window': 5,
}

# =============================================================================
# 드리블 점수 평가 설정
# =============================================================================
SCORING_CONFIG = {
    # 4가지 기준 가중치 (합계 = 1.0)
    'weights': {
        'head_up': 0.25,
        'trunk_lean': 0.25,
        'knee_updown': 0.25,
        'alignment': 0.25,
    },

    # 1. 헤드업 설정
    'head_up': {
        'touch_window': 5,            # 터치 전후 평가 프레임 수
        'optimal_angle_range': (-10, 10),  # 최적 시선 각도 (수평 기준)
        'worst_angle': 30.0,          # 최악 각도 (이 이상이면 0점)
        'min_visibility': 0.5,
    },

    # 2. 상체 기울기 설정
    'trunk_lean': {
        'optimal_range': (70, 85),    # 최적 상체 각도 (도)
        'worst_low': 60,              # 최저 허용 각도
        'worst_high': 90,             # 최고 허용 각도
        'min_visibility': 0.5,
    },

    # 3. 무릎 업다운 설정
    'knee_updown': {
        'touch_window': 10,           # 터치 전후 분석 프레임 수
        'min_angle_change': 10.0,     # 패턴 인정 최소 각도 변화
        'symmetry_bonus': 1.0,        # 양쪽 대칭시 추가 점수
        'min_visibility': 0.5,
    },

    # 4. 정렬 설정
    'alignment': {
        'optimal_angle': 15.0,        # 최적 정렬 각도 (이하면 10점)
        'worst_angle': 90.0,          # 최악 정렬 각도 (이상이면 0점)
        'velocity_window': 3,         # 볼 속도 계산 윈도우
        'min_velocity': 5.0,          # 정렬 평가할 최소 볼 속도
        'min_visibility': 0.5,
    },
}

# =============================================================================
# 레거시 호환성 (기존 SKILL_EVALUATION)
# =============================================================================
SKILL_EVALUATION = {
    'weights': {
        'head_up': 0.15,
        'torso_coordination': 0.40,
        'trunk_lean': 0.15,
        'knee_flexion': 0.20,
        'ankle_agility': 0.10
    },
    'head_up': {
        'optimal_range': (-10, 10),
        'worst': 30,
    },
    'torso_coordination': {
        'optimal_correlation': 0.95,
        'worst_correlation': 0.0,
    },
    'trunk_lean': {
        'optimal_range': (70, 85),
        'worst': 30,
    },
    'knee_flexion': {
        'optimal_angle': (120, 135),
        'worst_angle': 40,
        'optimal_rom': (90, 120),
        'worst_rom': 40,
    },
    'ankle_agility': {
        'optimal_speed': (0.5, 2.0),
        'worst_speed': 3.0,
        'optimal_acceleration': 33.0,
        'worst_acceleration': 0.0,
    }
}

# =============================================================================
# 출력 설정
# =============================================================================
OUTPUT_CONFIG = {
    'graphs_dir': 'output/graphs',
    'videos_dir': 'output/videos',
    'reports_dir': 'output/reports',
    'skeleton_color': (0, 255, 0),    # 초록색
    'ball_color': (0, 255, 255),      # 노란색
    'touch_color': (0, 0, 255),       # 빨간색
}
