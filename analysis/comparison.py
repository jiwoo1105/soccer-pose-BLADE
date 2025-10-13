# analysis/comparison.py
"""
=============================================================================
특성 기반 모션 비교 (프레임별 정렬 불필요)
=============================================================================

이 모듈은 두 드리블 모션의 통계적 특성을 비교합니다.

핵심 아이디어:
- 프레임 단위 비교가 아닌 "전체적인 패턴" 비교
- 두 비디오의 길이가 달라도 OK
- 평균, 범위, 분포 등 통계량으로 비교

비교 항목:
1. 평균값 (Mean): 전체 동작의 평균 자세
2. 범위 (Range): 움직임의 다이나믹함
3. 표준편차 (Std): 일관성
4. 최대/최소값: 극한 동작

출력:
1. DataFrame: 수치 비교 테이블
2. 텍스트 리포트: 사람이 읽을 수 있는 해석

예시:
    >>> comparator = MotionComparison(motion1, motion2, "선수A", "선수B")
    >>> table = comparator.compare_characteristics()
    >>> summary = comparator.generate_style_summary()
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class MotionComparison:
    """
    두 모션의 특성 비교 클래스

    역할:
    1. 각 motion_data에서 통계적 특성 추출 (평균, 범위 등)
    2. 두 통계를 비교하여 차이점 도출
    3. 사람이 이해하기 쉬운 해석 생성

    데이터 흐름:
        motion_data (각도 배열)
            ↓
        characteristics (통계량)
            ↓
        comparison_table (비교 결과)
            ↓
        style_summary (텍스트 리포트)
    """

    def __init__(self, motion1_data: Dict, motion2_data: Dict,
                 video1_name: str = "Video 1", video2_name: str = "Video 2"):
        """
        MotionComparison 초기화

        Args:
            motion1_data: 첫 번째 모션 데이터
                {
                    'left_knee_angles': np.array([152, 148, ...]),
                    'right_knee_angles': np.array([150, 145, ...]),
                    'trunk_angles': np.array([85, 84, ...]),
                    ...
                }
            motion2_data: 두 번째 모션 데이터 (동일한 구조)
            video1_name: 첫 번째 비디오 이름 (표시용)
            video2_name: 두 번째 비디오 이름 (표시용)

        사용 예시:
            >>> comparator = MotionComparison(
            >>>     motion1_data, motion2_data,
            >>>     "Soccer 1", "Soccer 2"
            >>> )
        """
        self.motion1 = motion1_data
        self.motion2 = motion2_data
        self.video1_name = video1_name
        self.video2_name = video2_name

    def extract_characteristics(self, motion_data: Dict, person_name: str) -> Dict:
        """
        드리블 특성 추출 - 각도 배열을 통계량으로 변환

        처리 과정:
        1. motion_data의 각 관절/분절 순회
        2. 각 배열에 대해 8가지 통계량 계산
        3. 딕셔너리로 반환

        Args:
            motion_data: 모션 데이터 (관절/분절별 각도 배열)
            person_name: 사람 이름 (현재는 미사용, 추후 확장용)

        Returns:
            Dict: 각 관절/분절의 통계적 특성
                {
                    'left_knee_angles': {
                        'mean': 150.2,      # 평균: 전체 평균 자세
                        'std': 5.3,         # 표준편차: 일관성 (낮을수록 일관됨)
                        'max': 165.1,       # 최댓값: 가장 많이 펴진 순간
                        'min': 135.7,       # 최솟값: 가장 많이 구부린 순간
                        'range': 29.4,      # 범위: 다이나믹함 (클수록 역동적)
                        'median': 149.8,    # 중앙값: 이상치에 강한 대표값
                        'percentile_25': 147.2,  # 하위 25%
                        'percentile_75': 153.5   # 상위 25%
                    },
                    'trunk_angles': {...},
                    ...
                }

        통계량의 의미:
            - mean: "평균적으로 어떤 자세인가"
            - range: "얼마나 다이나믹하게 움직이는가"
            - std: "얼마나 일정하게 움직이는가"
            - percentile: "대부분의 시간을 어떤 범위에서 보내는가"
        """
        characteristics = {}

        # STEP 1: motion_data의 각 항목 순회
        for joint_name, angles in motion_data.items():
            # 각도 배열이 아닌 다른 데이터는 스킵
            if not isinstance(angles, (list, np.ndarray)):
                continue

            # numpy 배열로 변환 (리스트일 수도 있으므로)
            angles = np.array(angles)

            # STEP 2: 8가지 통계량 계산
            characteristics[joint_name] = {
                'mean': float(np.mean(angles)),              # 평균
                'std': float(np.std(angles)),                # 표준편차
                'max': float(np.max(angles)),                # 최댓값
                'min': float(np.min(angles)),                # 최솟값
                'range': float(np.max(angles) - np.min(angles)),  # 범위
                'median': float(np.median(angles)),          # 중앙값
                'percentile_25': float(np.percentile(angles, 25)),  # 25% 지점
                'percentile_75': float(np.percentile(angles, 75))   # 75% 지점
            }

        return characteristics

    def compare_characteristics(self) -> pd.DataFrame:
        """
        두 모션의 특성 비교 - DataFrame으로 결과 반환

        처리 과정:
        1. 두 motion_data에서 각각 characteristics 추출
        2. 각 관절/분절별로 비교
        3. 차이값과 해석을 포함한 DataFrame 생성

        Returns:
            pd.DataFrame: 비교 결과 테이블
                | Joint/Segment    | Soccer1_Mean | Soccer2_Mean | Mean_Diff |
                |------------------|--------------|--------------|-----------|
                | left_knee_angles | 150.2        | 148.7        | 1.5       |
                | trunk_angles     | 85.2         | 88.1         | -2.9      |
                ...

        컬럼 설명:
            - Joint/Segment: 관절/분절 이름
            - {Video}_Mean: 각 비디오의 평균값
            - Mean_Diff: 평균값 차이 (양수 = Video1이 더 큼)
            - {Video}_ROM: Range of Motion (가동 범위)
            - ROM_Diff: 가동 범위 차이
            - {Video}_Min: 최솟값
            - Interpretation: 사람이 읽을 수 있는 해석

        활용:
            >>> table = comparator.compare_characteristics()
            >>> print(table.to_string())
            >>> table.to_csv('comparison.csv')
        """
        # STEP 1: 두 모션의 특성 추출
        char1 = self.extract_characteristics(self.motion1, self.video1_name)
        char2 = self.extract_characteristics(self.motion2, self.video2_name)

        # STEP 2: 비교 결과를 저장할 리스트
        comparison_results = []

        # STEP 3: 각 관절/분절에 대해 비교
        for joint in char1.keys():
            c1 = char1[joint]  # Video 1의 특성
            c2 = char2[joint]  # Video 2의 특성

            # 비교 결과 딕셔너리 생성
            result = {
                'Joint/Segment': joint,
                # 평균값 비교
                f'{self.video1_name}_Mean': round(c1['mean'], 1),
                f'{self.video2_name}_Mean': round(c2['mean'], 1),
                'Mean_Diff': round(c1['mean'] - c2['mean'], 1),
                # 가동 범위 비교 (다이나믹함)
                f'{self.video1_name}_ROM': round(c1['range'], 1),
                f'{self.video2_name}_ROM': round(c2['range'], 1),
                'ROM_Diff': round(c1['range'] - c2['range'], 1),
                # 최솟값 비교 (극한 동작)
                f'{self.video1_name}_Min': round(c1['min'], 1),
                f'{self.video2_name}_Min': round(c2['min'], 1),
                # 텍스트 해석
                'Interpretation': self._interpret_difference(joint, c1, c2)
            }

            comparison_results.append(result)

        # STEP 4: DataFrame으로 변환
        return pd.DataFrame(comparison_results)

    def _interpret_difference(self, joint: str, char1: Dict, char2: Dict) -> str:
        """
        차이를 사람이 이해하기 쉽게 해석 (내부 헬퍼 함수)

        룰 기반 해석:
        1. 평균 차이 해석
           - 10도 이상: "큰 차이"
           - 5-10도: "약간 차이"
           - 5도 미만: "유사"
        2. 가동 범위 차이 해석
           - 10도 이상: "더 다이나믹"
           - 5-10도: "약간 더 다이나믹"
           - 5도 미만: "유사"

        Args:
            joint: 관절/분절 이름
            char1: 첫 번째 특성
            char2: 두 번째 특성

        Returns:
            str: 해석 문자열
                예: "V1이 평균 12.3° 더 큼 (큰 차이), V2가 더 다이나믹"
        """
        # 평균 차이와 범위 차이 계산
        mean_diff = char1['mean'] - char2['mean']
        rom_diff = char1['range'] - char2['range']

        interpretation = []

        # PART 1: 평균 차이 해석
        if abs(mean_diff) > 10:
            # 큰 차이
            if mean_diff > 0:
                interpretation.append(f"V1이 평균 {abs(mean_diff):.1f}° 더 큼 (큰 차이)")
            else:
                interpretation.append(f"V2가 평균 {abs(mean_diff):.1f}° 더 큼 (큰 차이)")
        elif abs(mean_diff) > 5:
            # 약간 차이
            if mean_diff > 0:
                interpretation.append(f"V1이 평균 {abs(mean_diff):.1f}° 더 큼")
            else:
                interpretation.append(f"V2가 평균 {abs(mean_diff):.1f}° 더 큼")
        else:
            # 유사함
            interpretation.append("평균값 유사")

        # PART 2: 가동범위 차이 해석
        if abs(rom_diff) > 10:
            # 큰 차이
            if rom_diff > 0:
                interpretation.append(f"V1이 {abs(rom_diff):.1f}° 더 다이나믹")
            else:
                interpretation.append(f"V2가 {abs(rom_diff):.1f}° 더 다이나믹")
        elif abs(rom_diff) > 5:
            # 약간 차이
            if rom_diff > 0:
                interpretation.append(f"V1이 약간 더 다이나믹")
            else:
                interpretation.append(f"V2가 약간 더 다이나믹")
        else:
            # 유사함
            interpretation.append("가동범위 유사")

        # 두 해석을 쉼표로 연결
        return ", ".join(interpretation)

    def generate_style_summary(self) -> str:
        """
        전체 드리블 스타일 요약 - 텍스트 리포트 생성

        구성:
        1. 무릎 분석: 자세 높이 (Low/High stance)
        2. 몸통 분석: 전방 기울기 (공격성)
        3. 주요 발견사항: 가장 큰 차이 / 가장 유사한 부분
        4. 개선 제안: 차이가 큰 부분 나열

        Returns:
            str: 포맷된 텍스트 리포트

        출력 예시:
            ======================================================================
            DRIBBLING STYLE COMPARISON SUMMARY
            ======================================================================

            【 무릎 (Knee) 분석 】
            Soccer 1: High stance (안정적)
              - 평균 각도: 152.3°
              - 가동 범위: 45.2°
            Soccer 2: Low stance (공격적)
              - 평균 각도: 143.7°
              - 가동 범위: 52.1°
            → 차이: 8.6° (평균)

            【 몸통 (Trunk) 분석 】
            Soccer 1: 전방으로 4.8° 숙임
            Soccer 2: 전방으로 1.9° 숙임
            → 차이: 2.9°

            【 주요 발견사항 (Key Findings) 】
            ✓ 가장 큰 차이: left_knee_angles (8.6°)
            ✓ 가장 유사한 부분: trunk_angles (2.9°)

            【 개선 제안 (Recommendations) 】
            • Soccer 2: left_knee_angles을(를) 8.6° 더 크게
            ...
        """
        # STEP 1: 두 모션의 특성 추출
        char1 = self.extract_characteristics(self.motion1, self.video1_name)
        char2 = self.extract_characteristics(self.motion2, self.video2_name)

        # STEP 2: 리포트 헤더
        summary = "=" * 70 + "\n"
        summary += "DRIBBLING STYLE COMPARISON SUMMARY\n"
        summary += "=" * 70 + "\n\n"

        # SECTION 1: 무릎 분석
        if 'left_knee_angles' in char1:
            knee_avg1 = char1['left_knee_angles']['mean']
            knee_avg2 = char2['left_knee_angles']['mean']
            knee_rom1 = char1['left_knee_angles']['range']
            knee_rom2 = char2['left_knee_angles']['range']

            # 자세 스타일 판단
            # 145도 이하 = 낮은 자세 (공격적)
            # 145도 이상 = 높은 자세 (안정적)
            style1 = "Low stance (공격적)" if knee_avg1 < 145 else "High stance (안정적)"
            style2 = "Low stance (공격적)" if knee_avg2 < 145 else "High stance (안정적)"

            summary += f"【 무릎 (Knee) 분석 】\n"
            summary += f"{self.video1_name}: {style1}\n"
            summary += f"  - 평균 각도: {knee_avg1:.1f}°\n"
            summary += f"  - 가동 범위: {knee_rom1:.1f}°\n"
            summary += f"{self.video2_name}: {style2}\n"
            summary += f"  - 평균 각도: {knee_avg2:.1f}°\n"
            summary += f"  - 가동 범위: {knee_rom2:.1f}°\n"
            summary += f"→ 차이: {abs(knee_avg1 - knee_avg2):.1f}° (평균)\n\n"

        # SECTION 2: 몸통 분석
        if 'trunk_angles' in char1:
            trunk_avg1 = char1['trunk_angles']['mean']
            trunk_avg2 = char2['trunk_angles']['mean']

            # 전방 기울기 계산
            # trunk_angle이 90도면 수직 → 0도 숙임
            # trunk_angle이 85도면 → 5도 숙임
            lean1 = 90 - trunk_avg1
            lean2 = 90 - trunk_avg2

            summary += f"【 몸통 (Trunk) 분석 】\n"
            summary += f"{self.video1_name}: 전방으로 {lean1:.1f}° 숙임\n"
            summary += f"{self.video2_name}: 전방으로 {lean2:.1f}° 숙임\n"
            summary += f"→ 차이: {abs(lean1 - lean2):.1f}°\n\n"

        # SECTION 3: 주요 발견사항
        summary += "【 주요 발견사항 (Key Findings) 】\n"

        comparison_table = self.compare_characteristics()

        # 가장 큰 차이 찾기
        abs_diffs = comparison_table['Mean_Diff'].abs()
        max_diff_idx = abs_diffs.idxmax()
        max_diff_joint = comparison_table.loc[max_diff_idx, 'Joint/Segment']
        max_diff_value = comparison_table.loc[max_diff_idx, 'Mean_Diff']

        summary += f"✓ 가장 큰 차이: {max_diff_joint} ({abs(max_diff_value):.1f}°)\n"

        # 가장 유사한 부분 찾기
        min_diff_idx = abs_diffs.idxmin()
        min_diff_joint = comparison_table.loc[min_diff_idx, 'Joint/Segment']
        min_diff_value = comparison_table.loc[min_diff_idx, 'Mean_Diff']

        summary += f"✓ 가장 유사한 부분: {min_diff_joint} ({abs(min_diff_value):.1f}°)\n\n"

        # SECTION 4: 개선 제안
        summary += "【 개선 제안 (Recommendations) 】\n"

        # 5도 이상 차이나는 부분만 추출
        large_diffs = comparison_table[abs_diffs > 5]
        if len(large_diffs) > 0:
            for idx, row in large_diffs.iterrows():
                joint = row['Joint/Segment']
                diff = row['Mean_Diff']
                if diff > 0:
                    # Video1이 더 큼 → Video2가 개선할 점
                    summary += f"• {self.video2_name}: {joint}을(를) {abs(diff):.1f}° 더 크게\n"
                else:
                    # Video2가 더 큼 → Video1이 개선할 점
                    summary += f"• {self.video1_name}: {joint}을(를) {abs(diff):.1f}° 더 크게\n"
        else:
            summary += "• 두 영상의 동작이 전반적으로 유사합니다.\n"

        summary += "\n" + "=" * 70 + "\n"

        return summary

    def get_comparison_summary(self) -> Dict:
        """
        비교 요약을 딕셔너리로 반환 (프로그래밍 방식 접근용)

        텍스트 리포트 대신 구조화된 데이터로 반환
        다른 프로그램이나 웹 API에서 사용하기 좋음

        Returns:
            Dict: 비교 요약 정보
                {
                    'overall_similarity': 85.3,  # 0-100 점수
                    'most_different_joint': {
                        'name': 'left_knee_angles',
                        'difference': 8.6
                    },
                    'most_similar_joint': {
                        'name': 'trunk_angles',
                        'difference': 2.9
                    },
                    'significant_differences': [
                        {'joint': 'left_knee_angles', 'difference': 8.6},
                        {'joint': 'right_knee_angles', 'difference': 7.2}
                    ],
                    'comparison_table': DataFrame
                }
        """
        comparison_table = self.compare_characteristics()

        summary = {
            'overall_similarity': self._calculate_overall_similarity(comparison_table),
            'most_different_joint': None,
            'most_similar_joint': None,
            'significant_differences': [],
            'comparison_table': comparison_table
        }

        # 가장 다른 관절
        abs_diffs = comparison_table['Mean_Diff'].abs()
        max_diff_idx = abs_diffs.idxmax()
        summary['most_different_joint'] = {
            'name': comparison_table.loc[max_diff_idx, 'Joint/Segment'],
            'difference': float(comparison_table.loc[max_diff_idx, 'Mean_Diff'])
        }

        # 가장 유사한 관절
        min_diff_idx = abs_diffs.idxmin()
        summary['most_similar_joint'] = {
            'name': comparison_table.loc[min_diff_idx, 'Joint/Segment'],
            'difference': float(comparison_table.loc[min_diff_idx, 'Mean_Diff'])
        }

        # 유의미한 차이가 있는 관절들 (5도 이상)
        significant = comparison_table[abs_diffs > 5]
        summary['significant_differences'] = [
            {
                'joint': row['Joint/Segment'],
                'difference': float(row['Mean_Diff'])
            }
            for _, row in significant.iterrows()
        ]

        return summary

    def _calculate_overall_similarity(self, comparison_table: pd.DataFrame) -> float:
        """
        전체 유사도 계산 (0-100 점수)

        계산 방법:
        1. 모든 관절/분절의 평균 차이 절댓값 계산
        2. 선형 스케일링: 0도 차이 = 100점, 20도 차이 = 0점
        3. 100점 만점으로 변환

        Args:
            comparison_table: 비교 테이블

        Returns:
            float: 유사도 점수 (0-100)
                - 100: 완전히 동일
                - 90-100: 매우 유사
                - 70-90: 유사
                - 50-70: 약간 다름
                - 50 이하: 많이 다름

        예시:
            >>> similarity = comparator._calculate_overall_similarity(table)
            >>> print(f"유사도: {similarity:.1f}점")
        """
        # STEP 1: 평균 차이의 절댓값 평균
        mean_abs_diff = comparison_table['Mean_Diff'].abs().mean()

        # STEP 2: 선형 스케일링
        # 0도 차이 → 100점
        # 20도 차이 → 0점
        # 중간은 선형 보간
        similarity = max(0, 100 - (mean_abs_diff * 5))

        return round(similarity, 1)
