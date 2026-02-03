# scoring/dribble_scorer.py
"""
=============================================================================
드리블 종합 점수 평가 모듈
=============================================================================

4가지 평가 기준을 통합하여 종합 드리블 점수를 산출합니다.

평가 기준 (각 25%):
1. 헤드업 (Head Up): 터치 순간 시선이 수평인지
2. 상체 기울기 (Trunk Lean): 상체가 적절히 앞으로 기울어져 있는지
3. 무릎 업다운 (Knee Up-Down): 터치 순간 무릎 굽힘-펴짐 패턴
4. 정렬 (Alignment): 몸 방향과 볼 방향의 일치도

총점: 0-100점 스케일
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .head_up_scorer import HeadUpScorer, HeadUpScore
from .trunk_lean_scorer import TrunkLeanScorer, TrunkLeanScore
from .knee_updown_scorer import KneeUpDownScorer, KneeUpDownScore
from .alignment_scorer import AlignmentScorer, AlignmentScore


@dataclass
class DribbleScore:
    """드리블 종합 점수 데이터"""
    total_score: float  # 0-100 총점
    head_up_score: float  # 0-10 헤드업 점수
    trunk_lean_score: float  # 0-10 상체 기울기 점수
    knee_updown_score: float  # 0-10 무릎 업다운 점수
    alignment_score: float  # 0-10 정렬 점수
    head_up_detail: HeadUpScore  # 상세 헤드업 데이터
    trunk_lean_detail: TrunkLeanScore  # 상세 상체 기울기 데이터
    knee_updown_detail: KneeUpDownScore  # 상세 무릎 업다운 데이터
    alignment_detail: AlignmentScore  # 상세 정렬 데이터
    weights: Dict[str, float]  # 사용된 가중치
    grade: str  # 등급 (A, B, C, D, F)

    def __str__(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"드리블 종합 점수: {self.total_score:.1f}/100 ({self.grade})\n"
            f"{'='*60}\n"
            f"  헤드업:       {self.head_up_score:.1f}/10 (가중치: {self.weights['head_up']*100:.0f}%)\n"
            f"  상체 기울기:  {self.trunk_lean_score:.1f}/10 (가중치: {self.weights['trunk_lean']*100:.0f}%)\n"
            f"  무릎 업다운:  {self.knee_updown_score:.1f}/10 (가중치: {self.weights['knee_updown']*100:.0f}%)\n"
            f"  정렬:         {self.alignment_score:.1f}/10 (가중치: {self.weights['alignment']*100:.0f}%)\n"
            f"{'='*60}"
        )


# 기본 가중치
DEFAULT_WEIGHTS = {
    'head_up': 0.25,
    'trunk_lean': 0.25,
    'knee_updown': 0.25,
    'alignment': 0.25
}


class DribbleScorer:
    """
    드리블 종합 점수 평가기

    4가지 개별 평가 기준을 통합하여
    0-100점 스케일의 종합 점수를 산출합니다.

    사용 예시:
        >>> scorer = DribbleScorer()
        >>> result = scorer.score(pose_frames, touches, ball_data)
        >>> print(result)
        >>> print(f"등급: {result.grade}")
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 head_up_config: Optional[Dict] = None,
                 trunk_lean_config: Optional[Dict] = None,
                 knee_updown_config: Optional[Dict] = None,
                 alignment_config: Optional[Dict] = None):
        """
        DribbleScorer 초기화

        Args:
            weights: 각 기준별 가중치 (합계 1.0)
            head_up_config: 헤드업 평가기 설정
            trunk_lean_config: 상체 기울기 평가기 설정
            knee_updown_config: 무릎 업다운 평가기 설정
            alignment_config: 정렬 평가기 설정
        """
        # 가중치 설정
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self._validate_weights()

        # 개별 평가기 초기화
        self.head_up_scorer = HeadUpScorer(**(head_up_config or {}))
        self.trunk_lean_scorer = TrunkLeanScorer(**(trunk_lean_config or {}))
        self.knee_updown_scorer = KneeUpDownScorer(**(knee_updown_config or {}))
        self.alignment_scorer = AlignmentScorer(**(alignment_config or {}))

    def _validate_weights(self):
        """가중치 검증 및 정규화"""
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            print(f"Warning: 가중치 합계가 {total}입니다. 정규화합니다.")
            for key in self.weights:
                self.weights[key] /= total

    def score(self, pose_frames: List, touches: List = None,
              ball_data=None) -> DribbleScore:
        """
        드리블 종합 점수 계산

        Args:
            pose_frames: PoseFrame3D 또는 PoseFrame 리스트
            touches: TouchEvent 리스트 또는 터치 프레임 번호 리스트
            ball_data: BallMotionData 객체 (선택)

        Returns:
            DribbleScore: 종합 점수 데이터
        """
        # 터치 정보 없으면 ball_data에서 추출 시도
        if touches is None and ball_data is not None:
            if hasattr(ball_data, 'touch_frames'):
                touches = ball_data.touch_frames

        # 각 기준별 점수 계산
        head_up_result = self.head_up_scorer.score(pose_frames, touches or [])
        trunk_lean_result = self.trunk_lean_scorer.score(pose_frames)
        knee_updown_result = self.knee_updown_scorer.score(pose_frames, touches or [])
        alignment_result = self.alignment_scorer.score(pose_frames, ball_data)

        # 개별 점수 (0-10)
        head_up_score = head_up_result.total_score
        trunk_lean_score = trunk_lean_result.total_score
        knee_updown_score = knee_updown_result.total_score
        alignment_score = alignment_result.total_score

        # 가중 평균으로 총점 계산 (0-10)
        weighted_score = (
            head_up_score * self.weights['head_up'] +
            trunk_lean_score * self.weights['trunk_lean'] +
            knee_updown_score * self.weights['knee_updown'] +
            alignment_score * self.weights['alignment']
        )

        # 0-100 스케일로 변환
        total_score = weighted_score * 10

        # 등급 결정
        grade = self._calculate_grade(total_score)

        return DribbleScore(
            total_score=float(total_score),
            head_up_score=float(head_up_score),
            trunk_lean_score=float(trunk_lean_score),
            knee_updown_score=float(knee_updown_score),
            alignment_score=float(alignment_score),
            head_up_detail=head_up_result,
            trunk_lean_detail=trunk_lean_result,
            knee_updown_detail=knee_updown_result,
            alignment_detail=alignment_result,
            weights=self.weights.copy(),
            grade=grade
        )

    def _calculate_grade(self, score: float) -> str:
        """점수에 따른 등급 결정"""
        if score >= 90:
            return 'A+'
        elif score >= 80:
            return 'A'
        elif score >= 70:
            return 'B+'
        elif score >= 60:
            return 'B'
        elif score >= 50:
            return 'C+'
        elif score >= 40:
            return 'C'
        elif score >= 30:
            return 'D'
        else:
            return 'F'

    def score_frame_by_frame(self, pose_frames: List, touches: List = None,
                             ball_data=None) -> Dict[str, np.ndarray]:
        """
        프레임별 상세 점수 반환

        Returns:
            Dict with keys: 'head_up', 'trunk_lean', 'knee_updown', 'alignment'
            각 값은 프레임별 점수 배열
        """
        result = self.score(pose_frames, touches, ball_data)

        return {
            'head_up': result.head_up_detail.frame_scores,
            'trunk_lean': result.trunk_lean_detail.frame_scores,
            'knee_updown': np.array([]),  # 터치 기반이라 프레임별 점수 없음
            'alignment': result.alignment_detail.frame_scores
        }

    def get_improvement_suggestions(self, result: DribbleScore) -> List[str]:
        """
        개선 제안 생성

        Args:
            result: DribbleScore 객체

        Returns:
            List[str]: 개선 제안 목록
        """
        suggestions = []

        # 각 기준별 개선 제안
        if result.head_up_score < 7:
            mean_angle = result.head_up_detail.mean_gaze_angle
            if mean_angle > 15:
                suggestions.append(
                    f"헤드업: 터치 순간 고개를 더 들어주세요. "
                    f"현재 시선 각도가 평균 {mean_angle:.1f}도 아래를 향하고 있습니다."
                )
            else:
                suggestions.append(
                    "헤드업: 터치 순간에 전방을 주시하세요. 공을 보지 않아도 됩니다."
                )

        if result.trunk_lean_score < 7:
            mean_angle = result.trunk_lean_detail.mean_angle
            if mean_angle < 70:
                suggestions.append(
                    f"상체 기울기: 상체를 좀 더 앞으로 숙여주세요. "
                    f"현재 {mean_angle:.1f}도로 너무 서있습니다. (최적: 70-85도)"
                )
            elif mean_angle > 85:
                suggestions.append(
                    f"상체 기울기: 상체를 너무 많이 숙이지 마세요. "
                    f"현재 {mean_angle:.1f}도입니다. (최적: 70-85도)"
                )

        if result.knee_updown_score < 7:
            left_patterns = sum(result.knee_updown_detail.left_patterns)
            right_patterns = sum(result.knee_updown_detail.right_patterns)
            total_touches = len(result.knee_updown_detail.touch_scores)

            if left_patterns == 0 and right_patterns == 0:
                suggestions.append(
                    "무릎 업다운: 터치 순간 무릎을 굽혔다 펴는 동작을 해주세요. "
                    "탄력적인 무릎 사용이 필요합니다."
                )
            elif left_patterns != right_patterns:
                suggestions.append(
                    f"무릎 업다운: 양쪽 무릎을 더 대칭적으로 사용해주세요. "
                    f"(왼쪽 패턴: {left_patterns}/{total_touches}, "
                    f"오른쪽 패턴: {right_patterns}/{total_touches})"
                )

        if result.alignment_score < 7:
            mean_angle = result.alignment_detail.mean_angle
            suggestions.append(
                f"정렬: 몸을 볼 진행 방향으로 향하게 해주세요. "
                f"현재 평균 {mean_angle:.1f}도 벗어나 있습니다. (최적: 15도 이내)"
            )

        if not suggestions:
            if result.total_score >= 80:
                suggestions.append("훌륭합니다! 현재 자세를 유지하세요.")
            else:
                suggestions.append("전반적인 드리블 자세를 개선해보세요.")

        return suggestions

    def compare_scores(self, result1: DribbleScore, result2: DribbleScore) -> Dict:
        """
        두 점수 비교

        Returns:
            Dict: 비교 결과
        """
        comparison = {
            'total_diff': result2.total_score - result1.total_score,
            'head_up_diff': result2.head_up_score - result1.head_up_score,
            'trunk_lean_diff': result2.trunk_lean_score - result1.trunk_lean_score,
            'knee_updown_diff': result2.knee_updown_score - result1.knee_updown_score,
            'alignment_diff': result2.alignment_score - result1.alignment_score,
            'improved': result2.total_score > result1.total_score
        }
        return comparison
