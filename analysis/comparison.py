# analysis/comparison.py
"""
특성 기반 모션 비교 (프레임별 정렬 불필요)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


class MotionComparison:
    """
    두 모션의 특성 비교
    - 평균값 비교
    - 극값 비교  
    - 가동 범위 비교
    - 통계적 비교
    """
    
    def __init__(self, motion1_data: Dict, motion2_data: Dict, 
                 video1_name: str = "Video 1", video2_name: str = "Video 2"):
        """
        Args:
            motion1_data: {
                'left_knee_angles': [...],
                'right_knee_angles': [...],
                ...
            }
            motion2_data: 동일한 구조
            video1_name: 영상1 이름
            video2_name: 영상2 이름
        """
        self.motion1 = motion1_data
        self.motion2 = motion2_data
        self.video1_name = video1_name
        self.video2_name = video2_name
    
    def extract_characteristics(self, motion_data: Dict, person_name: str) -> Dict:
        """
        드리블 특성 추출
        
        Args:
            motion_data: 모션 데이터
            person_name: 사람 이름
            
        Returns:
            Dict: 각 관절/분절의 통계적 특성
        """
        characteristics = {}
        
        for joint_name, angles in motion_data.items():
            if not isinstance(angles, (list, np.ndarray)):
                continue
            
            angles = np.array(angles)
            
            characteristics[joint_name] = {
                'mean': float(np.mean(angles)),
                'std': float(np.std(angles)),
                'max': float(np.max(angles)),
                'min': float(np.min(angles)),
                'range': float(np.max(angles) - np.min(angles)),
                'median': float(np.median(angles)),
                'percentile_25': float(np.percentile(angles, 25)),
                'percentile_75': float(np.percentile(angles, 75))
            }
        
        return characteristics
    
    def compare_characteristics(self) -> pd.DataFrame:
        """
        특성 비교
        
        Returns:
            pd.DataFrame: 비교 결과 테이블
        """
        char1 = self.extract_characteristics(self.motion1, self.video1_name)
        char2 = self.extract_characteristics(self.motion2, self.video2_name)
        
        comparison_results = []
        
        for joint in char1.keys():
            c1 = char1[joint]
            c2 = char2[joint]
            
            result = {
                'Joint/Segment': joint,
                f'{self.video1_name}_Mean': round(c1['mean'], 1),
                f'{self.video2_name}_Mean': round(c2['mean'], 1),
                'Mean_Diff': round(c1['mean'] - c2['mean'], 1),
                f'{self.video1_name}_ROM': round(c1['range'], 1),
                f'{self.video2_name}_ROM': round(c2['range'], 1),
                'ROM_Diff': round(c1['range'] - c2['range'], 1),
                f'{self.video1_name}_Min': round(c1['min'], 1),
                f'{self.video2_name}_Min': round(c2['min'], 1),
                'Interpretation': self._interpret_difference(joint, c1, c2)
            }
            
            comparison_results.append(result)
        
        return pd.DataFrame(comparison_results)
    
    def _interpret_difference(self, joint: str, char1: Dict, char2: Dict) -> str:
        """
        차이를 사람이 이해하기 쉽게 해석
        
        Args:
            joint: 관절/분절 이름
            char1: 특성 1
            char2: 특성 2
            
        Returns:
            str: 해석 문자열
        """
        mean_diff = char1['mean'] - char2['mean']
        rom_diff = char1['range'] - char2['range']
        
        interpretation = []
        
        # 평균 차이 해석
        if abs(mean_diff) > 10:
            if mean_diff > 0:
                interpretation.append(f"V1이 평균 {abs(mean_diff):.1f}° 더 큼 (큰 차이)")
            else:
                interpretation.append(f"V2가 평균 {abs(mean_diff):.1f}° 더 큼 (큰 차이)")
        elif abs(mean_diff) > 5:
            if mean_diff > 0:
                interpretation.append(f"V1이 평균 {abs(mean_diff):.1f}° 더 큼")
            else:
                interpretation.append(f"V2가 평균 {abs(mean_diff):.1f}° 더 큼")
        else:
            interpretation.append("평균값 유사")
        
        # 가동범위 차이 해석
        if abs(rom_diff) > 10:
            if rom_diff > 0:
                interpretation.append(f"V1이 {abs(rom_diff):.1f}° 더 다이나믹")
            else:
                interpretation.append(f"V2가 {abs(rom_diff):.1f}° 더 다이나믹")
        elif abs(rom_diff) > 5:
            if rom_diff > 0:
                interpretation.append(f"V1이 약간 더 다이나믹")
            else:
                interpretation.append(f"V2가 약간 더 다이나믹")
        else:
            interpretation.append("가동범위 유사")
        
        return ", ".join(interpretation)
    
    def generate_style_summary(self) -> str:
        """
        전체 드리블 스타일 요약
        
        Returns:
            str: 스타일 요약 리포트
        """
        char1 = self.extract_characteristics(self.motion1, self.video1_name)
        char2 = self.extract_characteristics(self.motion2, self.video2_name)
        
        summary = "=" * 70 + "\n"
        summary += "DRIBBLING STYLE COMPARISON SUMMARY\n"
        summary += "=" * 70 + "\n\n"
        
        # 무릎 분석
        if 'left_knee_angles' in char1:
            knee_avg1 = char1['left_knee_angles']['mean']
            knee_avg2 = char2['left_knee_angles']['mean']
            knee_rom1 = char1['left_knee_angles']['range']
            knee_rom2 = char2['left_knee_angles']['range']
            
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
        
        # 몸통 분석
        if 'trunk_angles' in char1:
            trunk_avg1 = char1['trunk_angles']['mean']
            trunk_avg2 = char2['trunk_angles']['mean']
            
            # 90도에서 뺀 값 = 전방 기울기
            lean1 = 90 - trunk_avg1
            lean2 = 90 - trunk_avg2
            
            summary += f"【 몸통 (Trunk) 분석 】\n"
            summary += f"{self.video1_name}: 전방으로 {lean1:.1f}° 숙임\n"
            summary += f"{self.video2_name}: 전방으로 {lean2:.1f}° 숙임\n"
            summary += f"→ 차이: {abs(lean1 - lean2):.1f}°\n\n"
        
        # 주요 발견사항
        summary += "【 주요 발견사항 (Key Findings) 】\n"
        
        comparison_table = self.compare_characteristics()
        
        # 가장 큰 차이
        abs_diffs = comparison_table['Mean_Diff'].abs()
        max_diff_idx = abs_diffs.idxmax()
        max_diff_joint = comparison_table.loc[max_diff_idx, 'Joint/Segment']
        max_diff_value = comparison_table.loc[max_diff_idx, 'Mean_Diff']
        
        summary += f"✓ 가장 큰 차이: {max_diff_joint} ({abs(max_diff_value):.1f}°)\n"
        
        # 가장 유사한 부분
        min_diff_idx = abs_diffs.idxmin()
        min_diff_joint = comparison_table.loc[min_diff_idx, 'Joint/Segment']
        min_diff_value = comparison_table.loc[min_diff_idx, 'Mean_Diff']
        
        summary += f"✓ 가장 유사한 부분: {min_diff_joint} ({abs(min_diff_value):.1f}°)\n\n"
        
        # 추천 사항
        summary += "【 개선 제안 (Recommendations) 】\n"
        
        large_diffs = comparison_table[abs_diffs > 5]
        if len(large_diffs) > 0:
            for idx, row in large_diffs.iterrows():
                joint = row['Joint/Segment']
                diff = row['Mean_Diff']
                if diff > 0:
                    summary += f"• {self.video2_name}: {joint}을(를) {abs(diff):.1f}° 더 크게\n"
                else:
                    summary += f"• {self.video1_name}: {joint}을(를) {abs(diff):.1f}° 더 크게\n"
        else:
            summary += "• 두 영상의 동작이 전반적으로 유사합니다.\n"
        
        summary += "\n" + "=" * 70 + "\n"
        
        return summary
    
    def get_comparison_summary(self) -> Dict:
        """
        비교 요약을 딕셔너리로 반환
        
        Returns:
            Dict: 비교 요약 정보
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
        
        # 유의미한 차이가 있는 관절들
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
        전체 유사도 계산 (0-100)
        
        Args:
            comparison_table: 비교 테이블
            
        Returns:
            float: 유사도 점수
        """
        # 평균 차이의 절대값 평균
        mean_abs_diff = comparison_table['Mean_Diff'].abs().mean()
        
        # 차이가 작을수록 높은 점수
        # 0도 차이 = 100점, 20도 차이 = 0점
        similarity = max(0, 100 - (mean_abs_diff * 5))
        
        return round(similarity, 1)