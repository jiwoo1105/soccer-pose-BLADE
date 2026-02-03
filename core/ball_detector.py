# core/ball_detector.py
"""
=============================================================================
하이브리드 공 탐지 모듈 - YOLOv9 + 스켈레톤 폴백 + Kalman 필터
=============================================================================

이 모듈은 두 가지 방법을 조합하여 축구공을 추적합니다:

1. YOLOv9 (1순위): 높은 정확도의 객체 탐지
2. 스켈레톤 기반 추정 (2순위): YOLO 실패시 발 위치 기반 추정
3. Kalman 필터: 시간적 스무딩과 예측

장점:
- 공이 가려지거나 YOLO가 실패해도 추적 유지
- 물리적으로 합리적인 위치 추정
- 부드러운 궤적 생성
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List, Deque
from collections import deque

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False


class HybridBallDetector:
    """
    하이브리드 공 탐지기 - YOLOv9 + 스켈레톤 폴백 + Kalman 필터

    사용 예시:
        >>> detector = HybridBallDetector()
        >>> result = detector.detect_ball_with_box(frame, pose_frame)
        >>> if result:
        >>>     center_x, center_y, x1, y1, x2, y2 = result
    """

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 confidence: float = 0.3,
                 max_distance_threshold: float = 150.0,
                 history_size: int = 5,
                 smoothing_alpha: float = 0.3,
                 use_kalman: bool = True,
                 use_skeleton_fallback: bool = True):
        """
        HybridBallDetector 초기화

        Args:
            model_name: YOLO 모델 파일명 (yolov9c.pt 권장)
            confidence: 탐지 신뢰도 임계값
            max_distance_threshold: 이전 위치로부터 최대 허용 거리 (pixels)
            history_size: 이전 위치 히스토리 크기
            smoothing_alpha: EMA 스무딩 계수 (0~1)
            use_kalman: Kalman 필터 사용 여부
            use_skeleton_fallback: 스켈레톤 기반 폴백 사용 여부
        """
        print(f"YOLO 모델 로딩 중: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.ball_class_id = 32  # COCO dataset sports ball

        self.max_distance_threshold = max_distance_threshold
        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha

        # 이전 위치 히스토리
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=history_size)

        # EMA 스무딩 위치
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None

        # Kalman 필터
        self.use_kalman = use_kalman and KALMAN_AVAILABLE
        self.kalman = None
        if self.use_kalman:
            self._init_kalman()

        # 스켈레톤 폴백
        self.use_skeleton_fallback = use_skeleton_fallback

        # 탐지 상태
        self.frame_count = 0
        self.consecutive_misses = 0
        self.detection_source = None  # 'yolo', 'skeleton', 'kalman'

    def _init_kalman(self):
        """Kalman 필터 초기화 (2D 위치 + 속도)"""
        self.kalman = KalmanFilter(dim_x=4, dim_z=2)

        # 상태 전이 행렬 (위치 + 속도 모델)
        self.kalman.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])

        # 측정 행렬 (위치만 관측)
        self.kalman.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # 프로세스 노이즈 (가속도 불확실성)
        self.kalman.Q = np.eye(4) * 0.1

        # 측정 노이즈
        self.kalman.R = np.eye(2) * 5.0

        # 초기 공분산
        self.kalman.P = np.eye(4) * 100

        # 초기 상태
        self.kalman.x = np.zeros(4)

        self.kalman_initialized = False

    def _update_kalman(self, x: float, y: float):
        """Kalman 필터 업데이트"""
        if not self.use_kalman or self.kalman is None:
            return

        measurement = np.array([x, y])

        if not self.kalman_initialized:
            self.kalman.x = np.array([x, y, 0, 0])
            self.kalman_initialized = True
        else:
            self.kalman.predict()
            self.kalman.update(measurement)

    def _predict_kalman(self) -> Optional[Tuple[float, float]]:
        """Kalman 필터로 다음 위치 예측"""
        if not self.use_kalman or self.kalman is None or not self.kalman_initialized:
            return None

        self.kalman.predict()
        return (self.kalman.x[0], self.kalman.x[1])

    def _filter_by_color(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """주황색 꼬깔콘 제거"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return True

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 주황색 범위
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])

        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_ratio = np.sum(mask > 0) / mask.size

        # 50% 이상이 주황색이면 꼬깔콘
        return orange_ratio < 0.5

    def _filter_by_size(self, x1: int, y1: int, x2: int, y2: int, frame_height: int) -> bool:
        """크기와 종횡비로 필터링"""
        width = x2 - x1
        height = y2 - y1

        # 종횡비: 공은 거의 정사각형
        aspect_ratio = width / (height + 1e-6)
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            return False

        # 화면 대비 크기: 공은 화면의 3~15%
        relative_size = height / frame_height
        if relative_size < 0.03 or relative_size > 0.15:
            return False

        return True

    def _get_all_ball_candidates(self, frame: np.ndarray) -> List[Tuple[int, int, float, int, int, int, int]]:
        """YOLO로 탐지된 공 후보들 반환"""
        candidates = []
        frame_height = frame.shape[0]

        results = self.model(frame, verbose=False, conf=self.confidence)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                if class_id == self.ball_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 크기 필터링
                    if not self._filter_by_size(x1, y1, x2, y2, frame_height):
                        continue

                    # 색상 필터링 (꼬깔콘 제거)
                    if not self._filter_by_color(frame, x1, y1, x2, y2):
                        continue

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    conf = float(box.conf[0])

                    candidates.append((center_x, center_y, conf, x1, y1, x2, y2))

        return candidates

    def _select_best_candidate(self, candidates: List[Tuple[int, int, float, int, int, int, int]]) -> Optional[Tuple[int, int, int, int, int, int]]:
        """여러 공 후보 중 가장 적합한 것 선택"""
        if not candidates:
            return None

        # 히스토리가 있으면: 이전 위치와 가장 가까운 것
        if len(self.position_history) > 0:
            weights = np.exp(np.linspace(-1, 0, len(self.position_history)))
            weights /= weights.sum()

            expected_x = sum(w * x for w, (x, y) in zip(weights, self.position_history))
            expected_y = sum(w * y for w, (x, y) in zip(weights, self.position_history))

            best_candidate = None
            min_distance = float('inf')

            for cx, cy, conf, x1, y1, x2, y2 in candidates:
                distance = np.sqrt((cx - expected_x)**2 + (cy - expected_y)**2)

                if distance > self.max_distance_threshold:
                    continue

                if distance < min_distance:
                    min_distance = distance
                    best_candidate = (cx, cy, x1, y1, x2, y2)

            return best_candidate

        # 히스토리 없으면: 가장 높은 신뢰도
        else:
            best = max(candidates, key=lambda c: c[2])
            cx, cy, conf, x1, y1, x2, y2 = best
            return (cx, cy, x1, y1, x2, y2)

    def _estimate_from_skeleton(self, pose_frame) -> Optional[Tuple[int, int]]:
        """
        스켈레톤 기반 공 위치 추정

        발 위치를 기반으로 공의 예상 위치를 계산합니다.
        드리블 중에는 공이 발 앞에 있다고 가정합니다.
        """
        if pose_frame is None:
            return None

        # 발 위치 추출
        left_ankle = None
        right_ankle = None

        # PoseFrame3D (SMPL-X)
        if hasattr(pose_frame, 'joints_2d') and pose_frame.joints_2d is not None:
            # SMPL-X: left_ankle=7, right_ankle=8
            left_ankle = pose_frame.joints_2d[7]
            right_ankle = pose_frame.joints_2d[8]
        # PoseFrame (MediaPipe)
        elif hasattr(pose_frame, 'landmarks') and pose_frame.landmarks is not None:
            # MediaPipe: LEFT_ANKLE=27, RIGHT_ANKLE=28
            left_ankle = pose_frame.landmarks[27][:2]
            right_ankle = pose_frame.landmarks[28][:2]

        if left_ankle is None or right_ankle is None:
            return None

        # 더 낮은 발 (활성 발) 선택
        if left_ankle[1] > right_ankle[1]:  # y가 클수록 아래쪽
            active_foot = left_ankle
        else:
            active_foot = right_ankle

        # 발 앞쪽에 공 위치 추정 (정규화 좌표)
        estimated_x = active_foot[0]
        estimated_y = active_foot[1] + 0.05  # 발 아래쪽 (y축 방향)

        # 픽셀 좌표로 변환
        if hasattr(pose_frame, 'frame_width') and pose_frame.frame_width > 0:
            pixel_x = int(estimated_x * pose_frame.frame_width)
            pixel_y = int(estimated_y * pose_frame.frame_height)
            return (pixel_x, pixel_y)

        return None

    def _apply_smoothing(self, x: int, y: int) -> Tuple[int, int]:
        """EMA 스무딩 적용"""
        if self.smoothed_x is None or self.smoothed_y is None:
            self.smoothed_x = float(x)
            self.smoothed_y = float(y)
        else:
            self.smoothed_x = self.smoothing_alpha * x + (1 - self.smoothing_alpha) * self.smoothed_x
            self.smoothed_y = self.smoothing_alpha * y + (1 - self.smoothing_alpha) * self.smoothed_y

        return int(self.smoothed_x), int(self.smoothed_y)

    def detect_ball_with_box(self, frame: np.ndarray,
                              pose_frame=None) -> Optional[Tuple[int, int, int, int, int, int]]:
        """
        프레임에서 공의 중심 좌표와 바운딩 박스 탐지

        Args:
            frame: BGR 이미지
            pose_frame: PoseFrame3D 또는 PoseFrame (스켈레톤 폴백용)

        Returns:
            (center_x, center_y, x1, y1, x2, y2) 또는 None
        """
        self.frame_count += 1

        # STEP 1: YOLO로 공 탐지
        candidates = self._get_all_ball_candidates(frame)
        result = self._select_best_candidate(candidates)

        if result is not None:
            self.detection_source = 'yolo'
            self.consecutive_misses = 0
            center_x, center_y, x1, y1, x2, y2 = result

            # Kalman 필터 업데이트
            self._update_kalman(center_x, center_y)

            # 스무딩 적용
            smoothed_x, smoothed_y = self._apply_smoothing(center_x, center_y)

            # 히스토리 업데이트
            self.position_history.append((smoothed_x, smoothed_y))

            return (smoothed_x, smoothed_y, x1, y1, x2, y2)

        # STEP 2: YOLO 실패 - 스켈레톤 폴백
        self.consecutive_misses += 1

        if self.use_skeleton_fallback and pose_frame is not None:
            skeleton_pos = self._estimate_from_skeleton(pose_frame)
            if skeleton_pos is not None:
                self.detection_source = 'skeleton'
                center_x, center_y = skeleton_pos

                # Kalman 업데이트 (낮은 가중치)
                if self.use_kalman and self.kalman is not None:
                    # 스켈레톤 추정은 노이즈가 크므로 측정 노이즈 증가
                    old_R = self.kalman.R.copy()
                    self.kalman.R = np.eye(2) * 50.0
                    self._update_kalman(center_x, center_y)
                    self.kalman.R = old_R

                # 스무딩
                smoothed_x, smoothed_y = self._apply_smoothing(center_x, center_y)

                # 히스토리 업데이트
                self.position_history.append((smoothed_x, smoothed_y))

                # 가상 바운딩 박스 (30x30 픽셀)
                return (smoothed_x, smoothed_y,
                        smoothed_x - 15, smoothed_y - 15,
                        smoothed_x + 15, smoothed_y + 15)

        # STEP 3: Kalman 예측 사용 (마지막 수단)
        if self.use_kalman and self.consecutive_misses < 10:
            kalman_pos = self._predict_kalman()
            if kalman_pos is not None:
                self.detection_source = 'kalman'
                center_x, center_y = int(kalman_pos[0]), int(kalman_pos[1])

                # 히스토리 업데이트
                self.position_history.append((center_x, center_y))

                return (center_x, center_y,
                        center_x - 15, center_y - 15,
                        center_x + 15, center_y + 15)

        # 탐지 실패
        self.detection_source = None
        return None

    def get_detection_source(self) -> Optional[str]:
        """마지막 탐지 소스 반환 ('yolo', 'skeleton', 'kalman', None)"""
        return self.detection_source


# 레거시 호환성을 위한 별칭
BallDetector = HybridBallDetector
