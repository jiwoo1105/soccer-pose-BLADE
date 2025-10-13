# utils/math_utils.py
"""
=============================================================================
수학 유틸리티 함수들
=============================================================================

이 모듈은 3D 공간에서의 각도 계산을 위한 핵심 함수들을 제공합니다.
주로 MediaPipe에서 추출한 3D 랜드마크 좌표를 사용하여:
1. 두 벡터 사이의 각도 계산
2. 벡터와 수직/수평축의 각도 계산
3. 세 점으로 정의된 관절 각도 계산
4. 시계열 데이터 스무딩

사용 예시:
    >>> p1 = np.array([1, 0, 0])
    >>> p2 = np.array([0, 0, 0])
    >>> p3 = np.array([0, 1, 0])
    >>> angle = calculate_angle(p1, p2, p3)  # 90도
"""

import numpy as np


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    두 벡터 사이의 각도 계산 (코사인 법칙 사용)

    동작 원리:
    1. 두 벡터를 정규화 (단위 벡터로 변환)
    2. 내적(dot product) 계산 → cos(θ) 값
    3. arccos으로 라디안 각도 계산
    4. 라디안을 도(degree)로 변환

    Args:
        v1: 벡터 1 (3D numpy array, 예: [x, y, z])
        v2: 벡터 2 (3D numpy array)

    Returns:
        float: 두 벡터 사이의 각도 (degrees, 0~180도)

    예시:
        >>> v1 = np.array([1, 0, 0])  # X축 방향
        >>> v2 = np.array([0, 1, 0])  # Y축 방향
        >>> angle_between_vectors(v1, v2)  # 90.0
    """
    # STEP 1: 벡터 정규화 (길이를 1로 만듦)
    # 1e-10을 더해서 0으로 나누기 방지
    v1_normalized = v1 / (np.linalg.norm(v1) + 1e-10)
    v2_normalized = v2 / (np.linalg.norm(v2) + 1e-10)

    # STEP 2: 내적 계산 → cos(θ) 값
    # dot(a, b) = |a| * |b| * cos(θ)
    # 정규화된 벡터이므로 |a| = |b| = 1 → dot(a, b) = cos(θ)
    cos_angle = np.dot(v1_normalized, v2_normalized)

    # STEP 3: 부동소수점 오차로 인한 범위 벗어남 방지
    # cos 값은 반드시 -1 ~ 1 사이여야 함
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # STEP 4: arccos로 각도(라디안) 계산
    angle_rad = np.arccos(cos_angle)

    # STEP 5: 라디안을 도(degree)로 변환
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def angle_with_vertical(vector: np.ndarray) -> float:
    """
    벡터와 수직축(Y축)의 각도 계산

    용도: 신체 분절이 수직축에서 얼마나 기울어졌는지 측정
    예시:
    - 몸통이 얼마나 앞으로 숙여졌는지 (trunk angle)
    - 허벅지가 수직에서 얼마나 벗어났는지 (thigh angle)

    MediaPipe 좌표계:
    - X축: 좌우 (왼쪽이 +)
    - Y축: 상하 (위가 -, 아래가 +) ← 주의! 일반적인 수학과 반대
    - Z축: 앞뒤 (카메라에서 멀어지는 게 +)

    Args:
        vector: 3D 벡터 (예: 어깨→엉덩이 벡터)

    Returns:
        float: 수직축과의 각도 (degrees)
        - 0도 = 완전히 수직
        - 90도 = 완전히 수평

    예시:
        >>> trunk_vector = shoulder_center - hip_center
        >>> angle = angle_with_vertical(trunk_vector)
        >>> if angle > 15:
        >>>     print("앞으로 많이 숙임")
    """
    # MediaPipe에서 Y축 아래 방향이 양수이므로 [0, -1, 0]
    vertical = np.array([0, -1, 0])  # 아래 방향 (MediaPipe 좌표계)
    return angle_between_vectors(vector, vertical)


def angle_with_horizontal(vector: np.ndarray) -> float:
    """
    벡터와 수평축(X-Z 평면)의 각도 계산

    용도: 주로 발 각도 측정
    예시: 발뒤꿈치→발끝 벡터가 수평에서 얼마나 올라갔는지/내려갔는지

    동작 원리:
    1. 벡터를 X-Z 평면에 투영 (Y 좌표를 0으로)
    2. 원래 벡터와 투영된 벡터의 각도 계산

    Args:
        vector: 3D 벡터 (예: 발뒤꿈치→발끝 벡터)

    Returns:
        float: 수평축과의 각도 (degrees)
        - 0도 = 완전히 수평
        - 90도 = 완전히 수직

    예시:
        >>> foot_vector = foot_index - heel
        >>> angle = angle_with_horizontal(foot_vector)
        >>> if angle > 30:
        >>>     print("발끝이 많이 올라감")
    """
    # STEP 1: X-Z 평면으로 투영 (Y=0으로 설정)
    # 원래 벡터: [x, y, z] → 투영: [x, 0, z]
    horizontal_vector = np.array([vector[0], 0, vector[2]])

    # STEP 2: 투영된 벡터가 영벡터인지 확인
    # (원래 벡터가 완전히 수직이면 투영 벡터의 길이가 0)
    if np.linalg.norm(horizontal_vector) < 1e-10:
        return 90.0  # 완전히 수직

    # STEP 3: 원래 벡터와 투영 벡터의 각도 = 수평축과의 각도
    return angle_between_vectors(vector, horizontal_vector)


def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    세 점으로 정의된 각도 계산 (p1-p2-p3, p2가 꼭지점)

    용도: 관절 각도 계산의 핵심 함수
    예시:
    - 무릎 각도: 엉덩이(p1) - 무릎(p2) - 발목(p3)
    - 고관절 각도: 어깨(p1) - 엉덩이(p2) - 무릎(p3)

    동작 원리:
    1. p2를 원점으로 하는 두 벡터 생성
       - v1 = p1 - p2 (p2에서 p1로)
       - v2 = p3 - p2 (p2에서 p3로)
    2. 두 벡터 사이의 각도 계산

    Args:
        p1: 첫 번째 점 (3D 좌표)
        p2: 중간 점 (꼭지점, 관절 위치)
        p3: 세 번째 점 (3D 좌표)

    Returns:
        float: 각도 (degrees)
        - 0도 = 완전히 접힘
        - 180도 = 완전히 펴짐
        - 90도 = 직각

    예시:
        >>> hip = landmarks[23]     # 엉덩이
        >>> knee = landmarks[25]    # 무릎 (꼭지점)
        >>> ankle = landmarks[27]   # 발목
        >>> knee_angle = calculate_angle(hip, knee, ankle)
        >>> if knee_angle < 135:
        >>>     print("무릎을 많이 구부림 (낮은 자세)")

    시각적 표현:
        p1 (엉덩이)
         \
          \  v1
           \
            p2 (무릎) ← 꼭지점, 각도를 재는 곳
           /
          /  v2
         /
        p3 (발목)

        측정하는 각도 = v1과 v2 사이의 각도
    """
    # STEP 1: p2를 원점으로 하는 벡터 생성
    v1 = p1 - p2  # p2에서 p1 방향
    v2 = p3 - p2  # p2에서 p3 방향

    # STEP 2: 두 벡터 사이의 각도 계산
    return angle_between_vectors(v1, v2)


def smooth_angles(angles: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    각도 시계열 데이터를 smoothing (이동 평균 필터)

    용도: MediaPipe 추정값의 노이즈 제거
    - 프레임마다 미세하게 떨리는 값들을 부드럽게 함
    - 전체적인 움직임 패턴은 유지하면서 노이즈만 제거

    동작 원리:
    - Moving Average: 주변 값들의 평균으로 대체
    - window_size=5라면, 앞뒤 2개씩 총 5개 값의 평균

    Args:
        angles: 각도 배열 (시계열 데이터)
                예: [150.2, 148.7, 149.1, 151.3, ...]
        window_size: 윈도우 크기 (홀수 권장, 기본값 5)
                    - 클수록 더 부드러워지지만 지연 증가
                    - 작을수록 원본에 가깝지만 노이즈 많음

    Returns:
        np.ndarray: smoothed 각도 배열 (원본과 같은 길이)

    예시:
        >>> raw_angles = np.array([150, 148, 152, 149, 151])  # 떨림 있음
        >>> smooth = smooth_angles(raw_angles, window_size=3)
        >>> # smooth = [149.3, 150.0, 149.7, 150.7, 150.0]  # 부드러워짐

    주의:
        - 데이터가 window_size보다 짧으면 smoothing 안 함
        - 배열 양 끝은 약간 부정확할 수 있음 (convolution 특성)
    """
    # STEP 1: 데이터가 너무 짧으면 smoothing 불가
    if len(angles) < window_size:
        return angles

    # STEP 2: 이동 평균 필터 커널 생성
    # 예: window_size=5 → kernel = [0.2, 0.2, 0.2, 0.2, 0.2]
    kernel = np.ones(window_size) / window_size

    # STEP 3: Convolution으로 이동 평균 계산
    # mode='same': 출력 길이를 입력과 동일하게 유지
    smoothed = np.convolve(angles, kernel, mode='same')

    return smoothed
