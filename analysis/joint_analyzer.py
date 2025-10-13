# analysis/joint_analyzer.py
"""
=============================================================================
관절(Joint) 각도 분석
=============================================================================

이 모듈은 관절이 얼마나 접히거나 펴져 있는지 분석합니다.

관절(Joint)이란?
- 세 개의 landmark로 정의되는 각도
- 예: 무릎 = 엉덩이-무릎-발목 (무릎이 꼭지점)

측정 방식:
- 세 점이 이루는 각도 측정
- 180도 = 완전히 펴짐
- 90도 = 직각으로 구부림
- 0도 = 완전히 접힘

분석하는 6개 관절:
1. Left/Right Knee (무릎): 엉덩이-무릎-발목 ← 자세 높이 판단
2. Left/Right Hip (고관절): 어깨-엉덩이-무릎 ← 다리 들어올림
3. Left/Right Ankle (발목): 무릎-발목-발끝

활용:
>>> analyzer = JointAnalyzer()
>>> knee_angle = analyzer.calculate_knee_angle(landmarks, 'left')
>>> if knee_angle < 135:
>>>     print("낮은 자세 (무릎을 많이 구부림)")
"""

import numpy as np
from typing import Dict
import sys
sys.path.append('..')
from utils.math_utils import calculate_angle


class JointAnalyzer:
    """
    관절 각도 분석 클래스

    역할: 신체 관절이 얼마나 구부러졌는지 계산

    관절 vs 분절의 차이:
    - 관절: 세 점이 이루는 각도 (예: 무릎이 얼마나 접혔는지)
    - 분절: 두 점을 잇는 선분의 기울기 (예: 허벅지가 얼마나 앞으로 나갔는지)

    관절 각도의 의미:
    - 무릎 각도가 작다 = 무릎을 많이 구부림 = 낮은 자세
    - 무릎 각도가 크다 = 다리를 펴고 있음 = 높은 자세
    """

    def calculate_knee_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        무릎 각도 계산 (hip - knee - ankle)

        측정 방법:
        1. 엉덩이(p1), 무릎(p2), 발목(p3) 좌표 가져오기
        2. 무릎을 꼭지점으로 하는 각도 계산
        3. 180도에 가까울수록 다리가 펴진 상태

        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'

        Returns:
            float: 무릎 각도 (degrees)
                  - 180도 = 완전히 펴진 상태 (직립)
                  - 150-165도 = 약간 구부림 (일반적 드리블)
                  - 135도 이하 = 많이 구부림 (낮은 자세, 공격적)
                  - 90도 = 직각 (스쿼트)

        드리블 자세 판단:
            >>> knee_angle = analyzer.calculate_knee_angle(landmarks, 'left')
            >>> if knee_angle < 135:
            >>>     style = "Low stance (공격적)"
            >>> elif knee_angle < 155:
            >>>     style = "Medium stance"
            >>> else:
            >>>     style = "High stance (안정적)"

        시각적 표현:
            엉덩이 (hip)
               \
                \  v1
                 \
                  무릎 (knee) ← 꼭지점, 여기서 각도 측정
                 /
                /  v2
               /
            발목 (ankle)

            측정 각도 = v1과 v2 사이의 각도
        """
        # 좌우 선택
        if side == 'left':
            hip = landmarks[23]     # 왼쪽 엉덩이
            knee = landmarks[25]    # 왼쪽 무릎 (꼭지점)
            ankle = landmarks[27]   # 왼쪽 발목
        else:
            hip = landmarks[24]     # 오른쪽 엉덩이
            knee = landmarks[26]    # 오른쪽 무릎 (꼭지점)
            ankle = landmarks[28]   # 오른쪽 발목

        # 세 점으로 각도 계산 (무릎이 꼭지점)
        angle = calculate_angle(hip, knee, ankle)

        return angle

    def calculate_hip_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        고관절 각도 계산 (shoulder - hip - knee)

        측정 방법:
        - 어깨-엉덩이-무릎이 이루는 각도
        - 다리를 앞으로 들어올릴수록 각도가 작아짐

        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'

        Returns:
            float: 고관절 각도 (degrees)
                  - 180도 = 다리가 완전히 아래로 (차렷 자세)
                  - 150-170도 = 일반적인 걸음
                  - 135도 이하 = 다리를 많이 들어올림 (킥, 점프)

        활용:
            >>> hip_angle = analyzer.calculate_hip_angle(landmarks, 'left')
            >>> if hip_angle < 140:
            >>>     print("왼쪽 다리를 앞으로 많이 들어올림")

        시각적 표현:
            어깨 (shoulder)
               |
               |
            엉덩이 (hip) ← 꼭지점
              /
             /
            무릎 (knee)
        """
        if side == 'left':
            shoulder = landmarks[11]  # 왼쪽 어깨
            hip = landmarks[23]       # 왼쪽 엉덩이 (꼭지점)
            knee = landmarks[25]      # 왼쪽 무릎
        else:
            shoulder = landmarks[12]  # 오른쪽 어깨
            hip = landmarks[24]       # 오른쪽 엉덩이 (꼭지점)
            knee = landmarks[26]      # 오른쪽 무릎

        # 세 점으로 각도 계산 (엉덩이가 꼭지점)
        angle = calculate_angle(shoulder, hip, knee)

        return angle

    def calculate_ankle_angle(self, landmarks: np.ndarray, side: str = 'left') -> float:
        """
        발목 각도 계산 (knee - ankle - foot)

        측정 방법:
        - 무릎-발목-발끝이 이루는 각도
        - 발끝을 올릴수록(dorsiflexion) 각도가 작아짐
        - 발끝을 내릴수록(plantar flexion) 각도가 커짐

        Args:
            landmarks: (33, 3) landmarks array
            side: 'left' or 'right'

        Returns:
            float: 발목 각도 (degrees)
                  - 90도 = 발이 정강이에 수직 (일반적)
                  - 70도 = 발끝을 올림 (dorsiflexion)
                  - 110도 = 발끝을 내림 (plantar flexion, 발레)

        활용:
            >>> ankle_angle = analyzer.calculate_ankle_angle(landmarks, 'left')
            >>> if ankle_angle < 80:
            >>>     print("발목을 많이 구부림")

        시각적 표현:
            무릎 (knee)
              |
              |
            발목 (ankle) ← 꼭지점
             /
            /
          발끝 (foot_index)
        """
        if side == 'left':
            knee = landmarks[25]    # 왼쪽 무릎
            ankle = landmarks[27]   # 왼쪽 발목 (꼭지점)
            foot = landmarks[31]    # 왼쪽 발끝 (foot index)
        else:
            knee = landmarks[26]    # 오른쪽 무릎
            ankle = landmarks[28]   # 오른쪽 발목 (꼭지점)
            foot = landmarks[32]    # 오른쪽 발끝 (foot index)

        # 세 점으로 각도 계산 (발목이 꼭지점)
        angle = calculate_angle(knee, ankle, foot)

        return angle

    def calculate_all_joints(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        모든 관절 각도를 한번에 계산

        효율성: 각 관절을 개별적으로 호출하는 것보다 한번에 계산

        Args:
            landmarks: (33, 3) landmarks array

        Returns:
            Dict[str, float]: 각 관절의 각도를 담은 딕셔너리
                {
                    'left_knee': 152.3,
                    'right_knee': 148.7,
                    'left_hip': 165.2,
                    'right_hip': 162.8,
                    'left_ankle': 95.3,
                    'right_ankle': 92.1
                }

        활용:
            main.py에서 각 프레임마다 이 함수를 호출하여
            모든 관절 각도를 추출합니다.

        예시:
            >>> analyzer = JointAnalyzer()
            >>> joints = analyzer.calculate_all_joints(landmarks)
            >>> print(f"왼쪽 무릎 각도: {joints['left_knee']:.1f}도")
            >>> print(f"오른쪽 무릎 각도: {joints['right_knee']:.1f}도")
        """
        joints = {
            'left_knee': self.calculate_knee_angle(landmarks, 'left'),
            'right_knee': self.calculate_knee_angle(landmarks, 'right'),
            'left_hip': self.calculate_hip_angle(landmarks, 'left'),
            'right_hip': self.calculate_hip_angle(landmarks, 'right'),
            'left_ankle': self.calculate_ankle_angle(landmarks, 'left'),
            'right_ankle': self.calculate_ankle_angle(landmarks, 'right')
        }

        return joints
