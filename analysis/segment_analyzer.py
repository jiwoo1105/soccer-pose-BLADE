# analysis/segment_analyzer.py
"""
=============================================================================
Body Segment (신체 분절) 각도 분석
=============================================================================

이 모듈은 신체 분절(Body Segment)이 공간에서 얼마나 기울어져 있는지 분석합니다.

분절(Segment)이란?
- 두 관절을 연결하는 신체 부위
- 예: 엉덩이-무릎 = 허벅지(thigh), 무릎-발목 = 정강이(shank)

측정 방식:
- 수직축 대비: 분절이 수직에서 얼마나 벗어났는지
- 수평축 대비: 분절이 수평에서 얼마나 올라갔는지/내려갔는지

분석하는 7개 분절:
1. Trunk (몸통): 어깨-엉덩이 ← 자세의 공격성 판단
2. Left/Right Thigh (허벅지): 엉덩이-무릎
3. Left/Right Shank (정강이): 무릎-발목
4. Left/Right Foot (발): 발뒤꿈치-발끝

활용:
>>> analyzer = SegmentAnalyzer()
>>> trunk_angle = analyzer.calculate_trunk_angle(landmarks)
>>> if trunk_angle > 15:
>>>     print("공격적인 자세 (앞으로 많이 숙임)")
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from utils.math_utils import angle_between_vectors, angle_with_vertical, angle_with_horizontal


class SegmentAnalyzer:
    """
    신체 분절(Body Segment) 각도 분석 클래스

    역할: 신체 각 부위가 수직/수평축에 대해 얼마나 기울어졌는지 계산

    분절 vs 관절의 차이:
    - 분절: 두 점을 잇는 선분의 기울기 (예: 허벅지가 얼마나 앞으로 나갔는지)
    - 관절: 세 점이 이루는 각도 (예: 무릎이 얼마나 접혔는지)
    """

    def calculate_trunk_angle(self, landmarks: np.ndarray) -> float:
        """
        몸통 각도 계산 (어깨 중심 - 엉덩이 중심과 수직축의 각도)

        측정 방법:
        1. 양쪽 어깨의 중심점 계산
        2. 양쪽 엉덩이의 중심점 계산
        3. 어깨 중심 → 엉덩이 중심 벡터 생성
        4. 이 벡터와 수직축의 각도 측정

        Args:
            landmarks: (33, 3) world landmarks array

        Returns:
            float: 몸통 각도 (degrees)
                  - 0도 = 완전히 수직 (직립)
                  - 10-15도 = 약간 숙임 (일반적 드리블)
                  - 20도 이상 = 많이 숙임 (공격적)

        예시:
            >>> trunk_angle = analyzer.calculate_trunk_angle(landmarks)
            >>> lean = 90 - trunk_angle  # 전방 기울기
            >>> print(f"앞으로 {lean:.1f}도 숙임")
        """
        # STEP 1: 양쪽 어깨 랜드마크 가져오기
        left_shoulder = landmarks[11]   # 왼쪽 어깨
        right_shoulder = landmarks[12]  # 오른쪽 어깨

        # STEP 2: 양쪽 엉덩이 랜드마크 가져오기
        left_hip = landmarks[23]   # 왼쪽 엉덩이
        right_hip = landmarks[24]  # 오른쪽 엉덩이

        # STEP 3: 어깨 중심점과 엉덩이 중심점 계산
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        # STEP 4: 몸통 벡터 생성 (어깨 → 엉덩이 방향은 아래쪽)
        # 실제로는 엉덩이 → 어깨 방향으로 계산 (위쪽 벡터)
        trunk_vector = shoulder_center - hip_center

        # STEP 5: 수직축과의 각도 계산
        angle = angle_with_vertical(trunk_vector)

        return angle

    def calculate_thigh_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        허벅지 각도 계산 (엉덩이 - 무릎과 수직축의 각도)

        측정 방법:
        - 엉덩이 → 무릎 벡터와 수직축의 각도

        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'

        Returns:
            float: 허벅지 각도 (degrees)
                  - 0도 = 허벅지가 완전히 수직 (차렷 자세)
                  - 30도 = 다리를 약간 앞으로 뻗음
                  - 60도 = 다리를 많이 앞으로 뻗음 (킥 동작)

        예시:
            >>> left_thigh = analyzer.calculate_thigh_angle(landmarks, 'left')
            >>> right_thigh = analyzer.calculate_thigh_angle(landmarks, 'right')
            >>> if abs(left_thigh - right_thigh) > 30:
            >>>     print("좌우 다리 위치 차이가 큼")
        """
        # 좌우 선택
        if side == 'left':
            hip = landmarks[23]   # 왼쪽 엉덩이
            knee = landmarks[25]  # 왼쪽 무릎
        else:
            hip = landmarks[24]   # 오른쪽 엉덩이
            knee = landmarks[26]  # 오른쪽 무릎

        # 허벅지 벡터 (엉덩이 → 무릎)
        thigh_vector = knee - hip

        # 수직축과의 각도
        angle = angle_with_vertical(thigh_vector)

        return angle

    def calculate_shank_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        정강이 각도 계산 (무릎 - 발목과 수직축의 각도)

        측정 방법:
        - 무릎 → 발목 벡터와 수직축의 각도

        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'

        Returns:
            float: 정강이 각도 (degrees)
                  - 0도 = 정강이가 완전히 수직
                  - 20도 = 약간 기울어짐
                  - 40도 이상 = 많이 기울어짐 (낮은 자세 또는 점프)
        """
        if side == 'left':
            knee = landmarks[25]   # 왼쪽 무릎
            ankle = landmarks[27]  # 왼쪽 발목
        else:
            knee = landmarks[26]   # 오른쪽 무릎
            ankle = landmarks[28]  # 오른쪽 발목

        # 정강이 벡터 (무릎 → 발목)
        shank_vector = ankle - knee

        # 수직축과의 각도
        angle = angle_with_vertical(shank_vector)

        return angle

    def calculate_foot_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        발 각도 계산 (발뒤꿈치 - 발끝과 수평축의 각도)

        측정 방법:
        - 발뒤꿈치 → 발끝 벡터와 수평축의 각도

        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'

        Returns:
            float: 발 각도 (degrees)
                  - 0도 = 발이 완전히 수평 (평평함)
                  - 양수 = 발끝이 올라감 (dorsiflexion)
                  - 음수 = 발끝이 내려감 (plantar flexion)

        예시:
            >>> foot_angle = analyzer.calculate_foot_angle(landmarks, 'left')
            >>> if foot_angle > 30:
            >>>     print("발끝을 많이 올림")
        """
        if side == 'left':
            heel = landmarks[29]        # 왼쪽 발뒤꿈치
            foot_index = landmarks[31]  # 왼쪽 발끝
        else:
            heel = landmarks[30]        # 오른쪽 발뒤꿈치
            foot_index = landmarks[32]  # 오른쪽 발끝

        # 발 벡터 (발뒤꿈치 → 발끝)
        foot_vector = foot_index - heel

        # 수평축과의 각도
        angle = angle_with_horizontal(foot_vector)

        return angle

    def calculate_all_segments(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        모든 분절 각도를 한번에 계산

        효율성: 각 분절을 개별적으로 호출하는 것보다 한번에 계산

        Args:
            landmarks: (33, 3) landmarks array

        Returns:
            Dict[str, float]: 각 분절의 각도를 담은 딕셔너리
                {
                    'trunk': 85.2,
                    'left_thigh': 35.1,
                    'right_thigh': 32.8,
                    'left_shank': 12.3,
                    'right_shank': 15.7,
                    'left_foot': 10.5,
                    'right_foot': 12.1
                }

        활용:
            main.py에서 각 프레임마다 이 함수를 호출하여
            모든 분절 각도를 추출합니다.

        예시:
            >>> analyzer = SegmentAnalyzer()
            >>> segments = analyzer.calculate_all_segments(landmarks)
            >>> print(f"몸통 각도: {segments['trunk']:.1f}도")
        """
        segments = {
            'trunk': self.calculate_trunk_angle(landmarks),
            'left_thigh': self.calculate_thigh_angle(landmarks, 'left'),
            'right_thigh': self.calculate_thigh_angle(landmarks, 'right'),
            'left_shank': self.calculate_shank_angle(landmarks, 'left'),
            'right_shank': self.calculate_shank_angle(landmarks, 'right'),
            'left_foot': self.calculate_foot_angle(landmarks, 'left'),
            'right_foot': self.calculate_foot_angle(landmarks, 'right')
        }

        return segments
