# visualization/skeleton_drawer.py
"""
=============================================================================
스켈레톤 시각화
=============================================================================

이 모듈은 MediaPipe 포즈 landmark를 비디오 프레임에 그립니다.

주요 기능:
1. 33개 landmark 점 그리기
2. landmark 간 연결선 그리기
3. 텍스트 오버레이 (비디오 이름 등)

MediaPipe 연결 구조:
- 몸통: 어깨-어깨, 어깨-엉덩이, 엉덩이-엉덩이
- 팔: 어깨-팔꿈치-손목
- 다리: 엉덩이-무릎-발목-발뒤꿈치-발끝

활용:
>>> drawer = SkeletonDrawer(color=(0, 255, 0))  # 초록색
>>> frame_with_skeleton = drawer.draw_skeleton(frame, landmarks)
"""

import cv2
import numpy as np
from typing import Tuple


class SkeletonDrawer:
    """
    MediaPipe 포즈를 영상에 그리는 클래스

    역할:
    1. Normalized landmarks (0-1)를 픽셀 좌표로 변환
    2. 관절 점과 연결선을 OpenCV로 그리기
    3. 텍스트 오버레이 추가

    좌표 변환:
        landmarks[i][0] (0-1) → x_pixel (0-width)
        landmarks[i][1] (0-1) → y_pixel (0-height)
    """

    # MediaPipe 연결선 정의 (landmark 인덱스 쌍)
    # 각 튜플은 (시작점, 끝점) 인덱스
    CONNECTIONS = [
        # 몸통 (Torso)
        (11, 12),  # 왼쪽 어깨 - 오른쪽 어깨
        (11, 23),  # 왼쪽 어깨 - 왼쪽 엉덩이
        (12, 24),  # 오른쪽 어깨 - 오른쪽 엉덩이
        (23, 24),  # 왼쪽 엉덩이 - 오른쪽 엉덩이

        # 왼쪽 팔 (Left Arm)
        (11, 13),  # 어깨 - 팔꿈치
        (13, 15),  # 팔꿈치 - 손목

        # 오른쪽 팔 (Right Arm)
        (12, 14),  # 어깨 - 팔꿈치
        (14, 16),  # 팔꿈치 - 손목

        # 왼쪽 다리 (Left Leg)
        (23, 25),  # 엉덩이 - 무릎
        (25, 27),  # 무릎 - 발목
        (27, 29),  # 발목 - 발뒤꿈치
        (29, 31),  # 발뒤꿈치 - 발끝

        # 오른쪽 다리 (Right Leg)
        (24, 26),  # 엉덩이 - 무릎
        (26, 28),  # 무릎 - 발목
        (28, 30),  # 발목 - 발뒤꿈치
        (30, 32),  # 발뒤꿈치 - 발끝
    ]

    def __init__(self, color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2,
                 point_radius: int = 4):
        """
        SkeletonDrawer 초기화

        Args:
            color: 선 색상 (BGR 형식)
                  예: (0, 255, 0) = 초록색
                      (255, 0, 0) = 파란색
                      (0, 0, 255) = 빨간색
            thickness: 선 두께 (픽셀)
            point_radius: 점 반지름 (픽셀)

        참고: OpenCV는 BGR 순서 사용 (RGB 아님!)

        사용 예시:
            >>> # 파란색 스켈레톤
            >>> drawer1 = SkeletonDrawer(color=(255, 0, 0))
            >>> # 빨간색 스켈레톤, 더 굵게
            >>> drawer2 = SkeletonDrawer(color=(0, 0, 255), thickness=3)
        """
        self.color = color
        self.thickness = thickness
        self.point_radius = point_radius

    def draw_skeleton(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        프레임에 스켈레톤 그리기

        처리 과정:
        1. 연결선 그리기: CONNECTIONS에 정의된 순서대로
        2. 관절점 그리기: 모든 33개 landmark
        3. 원본 프레임은 보존하고 복사본에 그림

        Args:
            frame: BGR 이미지 (OpenCV 형식)
                  shape: (height, width, 3)
            landmarks: (33, 3) normalized landmarks [0-1 범위]
                      landmarks[i] = [x, y, z]
                      - x, y는 0-1로 정규화됨
                      - z는 깊이 (여기서는 사용 안 함)

        Returns:
            np.ndarray: 스켈레톤이 그려진 이미지

        주의:
            - landmarks의 x, y가 0-1 범위가 아니면 화면 밖에 그려질 수 있음
            - 원본 frame은 수정되지 않음 (복사본 반환)

        예시:
            >>> drawer = SkeletonDrawer()
            >>> result = drawer.draw_skeleton(frame, landmarks)
            >>> cv2.imshow("Skeleton", result)
        """
        # STEP 1: 프레임 크기 가져오기
        h, w, _ = frame.shape

        # STEP 2: 원본 보존을 위해 복사
        output = frame.copy()

        # STEP 3: 연결선 그리기
        for start_idx, end_idx in self.CONNECTIONS:
            # 시작점과 끝점 landmark 가져오기
            start_point = landmarks[start_idx]  # [x, y, z] (0-1 범위)
            end_point = landmarks[end_idx]

            # Normalized coordinates (0-1)를 픽셀 좌표로 변환
            # x * width = 픽셀 x좌표
            # y * height = 픽셀 y좌표
            start_pixel = (int(start_point[0] * w), int(start_point[1] * h))
            end_pixel = (int(end_point[0] * w), int(end_point[1] * h))

            # OpenCV로 선 그리기
            cv2.line(output, start_pixel, end_pixel, self.color, self.thickness)

        # STEP 4: 관절점 그리기
        for landmark in landmarks:
            # 픽셀 좌표로 변환
            pixel = (int(landmark[0] * w), int(landmark[1] * h))

            # 원(circle) 그리기
            # -1: 채워진 원
            cv2.circle(output, pixel, self.point_radius, self.color, -1)

        return output

    def draw_angle_text(self, frame: np.ndarray,
                       text: str,
                       position: Tuple[int, int],
                       color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        프레임에 텍스트 그리기 (배경 박스 포함)

        용도:
        - 비디오 이름 표시
        - 각도 값 표시
        - 프레임 번호 표시 등

        특징:
        - 검은색 배경 박스로 가독성 향상
        - 흰색 텍스트 (기본값)

        Args:
            frame: BGR 이미지
            text: 표시할 텍스트
            position: (x, y) 텍스트 시작 위치 (픽셀)
            color: 텍스트 색상 (BGR)

        Returns:
            np.ndarray: 텍스트가 추가된 이미지

        예시:
            >>> drawer = SkeletonDrawer()
            >>> frame = drawer.draw_angle_text(
            >>>     frame, "Soccer 1", (10, 30)
            >>> )
            >>> frame = drawer.draw_angle_text(
            >>>     frame, "Knee: 152.3°", (10, 60), color=(0, 255, 0)
            >>> )
        """
        # STEP 1: 원본 보존을 위해 복사
        output = frame.copy()

        # STEP 2: 폰트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # STEP 3: 텍스트 크기 측정 (배경 박스 크기 계산용)
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # STEP 4: 배경 박스 그리기 (검은색)
        # 텍스트보다 약간 큰 박스
        cv2.rectangle(output,
                     (position[0] - 5, position[1] - text_h - 5),  # 좌상단
                     (position[0] + text_w + 5, position[1] + 5),  # 우하단
                     (0, 0, 0),  # 검은색
                     -1)  # 채워진 사각형

        # STEP 5: 텍스트 그리기
        cv2.putText(output, text, position, font, font_scale, color, thickness)

        return output
