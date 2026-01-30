# core/ball_detector.py
"""공 탐지 모듈 - YOLOv8 + 색상/크기 필터링 + 이동 평균 스무딩"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Tuple, List, Deque
from collections import deque


class BallDetector:

    def __init__(self,
                 model_name: str = 'yolov8n.pt',
                 confidence: float = 0.3,
                 max_distance_threshold: float = 150.0,
                 history_size: int = 5,
                 smoothing_alpha: float = 0.3):
        """
        BallDetector 초기화

        Args:
            model_name: YOLO 모델 파일명
            confidence: 탐지 신뢰도 임계값
            max_distance_threshold: 이전 위치로부터 최대 허용 거리 (pixels)
            history_size: 이전 위치 히스토리 크기
            smoothing_alpha: 스무딩 계수 (0~1, 클수록 새 값에 민감)
        """
        print(f"YOLOv8 모델 로딩 중: {model_name}")
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.ball_class_id = 32  # COCO dataset sports ball

        self.max_distance_threshold = max_distance_threshold
        self.history_size = history_size
        self.smoothing_alpha = smoothing_alpha

        # 이전 위치 히스토리
        self.position_history: Deque[Tuple[int, int]] = deque(maxlen=history_size)

        # 스무딩된 위치 (이동 평균)
        self.smoothed_x: Optional[float] = None
        self.smoothed_y: Optional[float] = None

        self.frame_count = 0

    def _filter_by_color(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> bool:
        """주황색 꼬깔콘 제거"""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return True

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 주황색만 체크
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
        """YOLOv8로 탐지된 공 후보들 반환 (색상/크기 필터링 적용)"""
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
        """여러 공 후보 중에서 가장 적합한 것을 선택"""
        if not candidates:
            return None

        # 히스토리가 있으면: 이전 위치와 가장 가까운 것 선택
        if len(self.position_history) > 0:
            # 최근 위치의 가중 평균 계산
            weights = np.exp(np.linspace(-1, 0, len(self.position_history)))
            weights /= weights.sum()

            expected_x = sum(w * x for w, (x, y) in zip(weights, self.position_history))
            expected_y = sum(w * y for w, (x, y) in zip(weights, self.position_history))

            # 예상 위치와 가장 가까운 후보 찾기
            best_candidate = None
            min_distance = float('inf')

            for cx, cy, conf, x1, y1, x2, y2 in candidates:
                distance = np.sqrt((cx - expected_x)**2 + (cy - expected_y)**2)

                # 거리 임계값 체크
                if distance > self.max_distance_threshold:
                    continue

                if distance < min_distance:
                    min_distance = distance
                    best_candidate = (cx, cy, x1, y1, x2, y2)

            return best_candidate

        # 히스토리가 없으면: 신뢰도가 가장 높은 후보 선택
        else:
            best = max(candidates, key=lambda c: c[2])
            cx, cy, conf, x1, y1, x2, y2 = best
            return (cx, cy, x1, y1, x2, y2)

    def _apply_smoothing(self, x: int, y: int) -> Tuple[int, int]:
        """
        간단한 지수 이동 평균(EMA)으로 스무딩

        smoothed = alpha × new_value + (1 - alpha) × smoothed
        - alpha가 클수록 새 값에 민감 (덜 부드러움)
        - alpha가 작을수록 이전 값 유지 (더 부드러움)
        """
        if self.smoothed_x is None or self.smoothed_y is None:
            # 첫 프레임
            self.smoothed_x = float(x)
            self.smoothed_y = float(y)
        else:
            # 지수 이동 평균
            self.smoothed_x = self.smoothing_alpha * x + (1 - self.smoothing_alpha) * self.smoothed_x
            self.smoothed_y = self.smoothing_alpha * y + (1 - self.smoothing_alpha) * self.smoothed_y

        return int(self.smoothed_x), int(self.smoothed_y)

    def detect_ball_with_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
        """프레임에서 공의 중심 좌표와 바운딩 박스를 탐지"""
        self.frame_count += 1

        # STEP 1: YOLO로 모든 공 후보 탐지
        candidates = self._get_all_ball_candidates(frame)

        # STEP 2: 가장 적합한 후보 선택
        result = self._select_best_candidate(candidates)

        if result is None:
            return None

        center_x, center_y, x1, y1, x2, y2 = result

        # STEP 3: 간단한 이동 평균으로 스무딩
        smoothed_x, smoothed_y = self._apply_smoothing(center_x, center_y)

        # STEP 4: 히스토리 업데이트
        self.position_history.append((smoothed_x, smoothed_y))

        return (smoothed_x, smoothed_y, x1, y1, x2, y2)
