# scoring/alignment_scorer.py
"""
=============================================================================
어깨-골반-볼 정렬(Alignment) 점수 평가 모듈
=============================================================================

어깨와 골반의 방향이 볼 이동 방향과 일치하는지 평가합니다.

규칙:
- 어깨/골반 방향이 볼 진행 방향과 일치해야 함
- 15도 이내면 10점
- 90도면 0점

몸 방향 계산:
- 어깨 벡터 (left_shoulder → right_shoulder)
- 수직 벡터와 외적하여 전방 방향 도출

점수 기준:
- 0-15도: 10점
- 15-90도: 선형 감점
- 90도 이상: 0점
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AlignmentScore:
    """정렬 점수 데이터"""
    total_score: float  # 0-10 점수
    frame_scores: np.ndarray  # 각 프레임별 점수
    frame_angles: np.ndarray  # 각 프레임별 정렬 각도
    mean_angle: float  # 평균 정렬 각도
    body_directions: List[np.ndarray]  # 몸 방향 벡터들
    ball_directions: List[np.ndarray]  # 볼 방향 벡터들


class AlignmentScorer:
    """
    어깨-골반-볼 정렬 점수 평가기

    몸의 전방 방향과 볼 이동 방향의 일치도를 평가합니다.
    3D 좌표를 사용하여 뷰 불변적인 점수를 산출합니다.

    사용 예시:
        >>> scorer = AlignmentScorer()
        >>> score = scorer.score(pose_frames, ball_data)
        >>> print(f"정렬 점수: {score.total_score:.1f}/10")
    """

    def __init__(self,
                 optimal_angle: float = 15.0,
                 worst_angle: float = 90.0,
                 velocity_window: int = 3,
                 min_velocity: float = 5.0,
                 min_visibility: float = 0.5):
        """
        AlignmentScorer 초기화

        Args:
            optimal_angle: 최적 정렬 각도 (이하면 10점)
            worst_angle: 최악 정렬 각도 (이상이면 0점)
            velocity_window: 볼 속도 계산 윈도우 (프레임)
            min_velocity: 정렬 평가할 최소 볼 속도 (pixels/frame)
            min_visibility: 최소 관절 신뢰도
        """
        self.optimal_angle = optimal_angle
        self.worst_angle = worst_angle
        self.velocity_window = velocity_window
        self.min_velocity = min_velocity
        self.min_visibility = min_visibility

    def score(self, pose_frames: List, ball_data=None) -> AlignmentScore:
        """
        포즈 프레임과 볼 데이터로 정렬 점수 계산

        Args:
            pose_frames: PoseFrame3D 또는 PoseFrame 리스트
            ball_data: BallMotionData 객체 또는 None

        Returns:
            AlignmentScore: 점수 데이터
        """
        if not pose_frames:
            return AlignmentScore(
                total_score=0.0,
                frame_scores=np.array([]),
                frame_angles=np.array([]),
                mean_angle=0.0,
                body_directions=[],
                ball_directions=[]
            )

        # 볼 속도 추출
        ball_velocities = self._extract_ball_velocities(pose_frames, ball_data)

        # 각 프레임별 정렬 점수 계산
        frame_scores = []
        frame_angles = []
        body_directions = []
        ball_directions = []

        for i, pf in enumerate(pose_frames):
            # 몸 전방 방향 계산
            body_forward = self._calculate_body_forward(pf)

            # 볼 이동 방향
            ball_velocity = ball_velocities.get(i) if ball_velocities else None

            if body_forward is not None and ball_velocity is not None:
                # 수평면 투영
                body_forward_h = self._project_to_horizontal(body_forward)
                ball_velocity_h = self._project_to_horizontal(ball_velocity)

                # 정렬 각도 계산
                angle = self._angle_between_vectors(body_forward_h, ball_velocity_h)

                # 점수 계산
                score = self._score_alignment(angle)

                frame_scores.append(score)
                frame_angles.append(angle)
                body_directions.append(body_forward_h)
                ball_directions.append(ball_velocity_h)
            else:
                # 데이터 부족
                frame_scores.append(np.nan)
                frame_angles.append(np.nan)
                body_directions.append(None)
                ball_directions.append(None)

        # 유효한 값으로 통계 계산
        valid_scores = [s for s in frame_scores if not np.isnan(s)]
        valid_angles = [a for a in frame_angles if not np.isnan(a)]

        if valid_scores:
            total_score = np.mean(valid_scores)
            mean_angle = np.mean(valid_angles)
        else:
            total_score = 5.0  # 데이터 부족시 중간 점수
            mean_angle = 45.0

        return AlignmentScore(
            total_score=float(total_score),
            frame_scores=np.array(frame_scores),
            frame_angles=np.array(frame_angles),
            mean_angle=float(mean_angle),
            body_directions=body_directions,
            ball_directions=ball_directions
        )

    def _calculate_body_forward(self, pose_frame) -> Optional[np.ndarray]:
        """
        몸의 전방 방향 계산

        어깨 벡터와 수직 벡터의 외적으로 전방 방향을 계산합니다.

        Returns:
            np.ndarray: 전방 방향 단위 벡터 (3D)
        """
        # PoseFrame3D (SMPL-X)
        if hasattr(pose_frame, 'joints_3d') and pose_frame.joints_3d is not None:
            joints = pose_frame.joints_3d
            confidence = pose_frame.confidence

            # SMPL-X: left_shoulder=16, right_shoulder=17
            if confidence[16] < self.min_visibility or confidence[17] < self.min_visibility:
                return None

            left_shoulder = joints[16]
            right_shoulder = joints[17]

            # 어깨 벡터: 왼쪽 → 오른쪽
            shoulder_vec = right_shoulder - left_shoulder

            # 수직 벡터
            vertical = np.array([0, 1, 0])

            # 전방 방향: 어깨 × 수직
            forward = np.cross(shoulder_vec, vertical)

            # 정규화
            norm = np.linalg.norm(forward)
            if norm < 1e-10:
                return None

            return forward / norm

        # PoseFrame (MediaPipe)
        elif hasattr(pose_frame, 'world_landmarks') and pose_frame.world_landmarks is not None:
            landmarks = pose_frame.world_landmarks
            visibility = pose_frame.visibility

            # MediaPipe: LEFT_SHOULDER=11, RIGHT_SHOULDER=12
            if visibility[11] < self.min_visibility or visibility[12] < self.min_visibility:
                return None

            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]

            shoulder_vec = right_shoulder - left_shoulder
            vertical = np.array([0, -1, 0])  # MediaPipe는 Y축이 반대

            forward = np.cross(shoulder_vec, vertical)

            norm = np.linalg.norm(forward)
            if norm < 1e-10:
                return None

            return forward / norm

        return None

    def _extract_ball_velocities(self, pose_frames: List, ball_data) -> Optional[dict]:
        """
        볼 속도 벡터 추출

        Returns:
            dict: {frame_idx: velocity_vector}
        """
        velocities = {}

        # 볼 위치 추출
        ball_positions = []
        for pf in pose_frames:
            if pf.ball_position is not None:
                ball_positions.append([pf.frame_number, pf.ball_position[0], pf.ball_position[1]])
            else:
                ball_positions.append([pf.frame_number, np.nan, np.nan])

        ball_positions = np.array(ball_positions)

        # 속도 계산 (이동 평균)
        for i in range(self.velocity_window, len(pose_frames)):
            start_idx = i - self.velocity_window
            end_idx = i

            # 윈도우 내 유효한 위치들
            window = ball_positions[start_idx:end_idx+1]
            valid_mask = ~np.isnan(window[:, 1])

            if np.sum(valid_mask) < 2:
                continue

            # 첫 번째와 마지막 유효 위치로 속도 계산
            valid_positions = window[valid_mask]
            dx = valid_positions[-1, 1] - valid_positions[0, 1]
            dy = valid_positions[-1, 2] - valid_positions[0, 2]
            dt = valid_positions[-1, 0] - valid_positions[0, 0]

            if dt > 0:
                velocity = np.array([dx / dt, 0, dy / dt])  # 2D를 3D로 (y=0)
                speed = np.linalg.norm(velocity)

                if speed >= self.min_velocity:
                    velocities[i] = velocity

        return velocities if velocities else None

    def _project_to_horizontal(self, vector: np.ndarray) -> np.ndarray:
        """벡터를 수평면(X-Z)에 투영"""
        if len(vector) < 3:
            return vector

        horizontal = np.array([vector[0], 0, vector[2]])
        norm = np.linalg.norm(horizontal)

        if norm < 1e-10:
            return np.array([1, 0, 0])

        return horizontal / norm

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """두 벡터 사이의 각도 (도)"""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)

        if v1_norm < 1e-10 or v2_norm < 1e-10:
            return 90.0

        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)

        return np.degrees(angle_rad)

    def _score_alignment(self, angle: float) -> float:
        """
        정렬 각도에 대한 점수

        Args:
            angle: 정렬 각도 (도)

        Returns:
            float: 0-10 점수
        """
        # 최적 범위 내: 10점
        if angle <= self.optimal_angle:
            return 10.0

        # 최악 각도 이상: 0점
        if angle >= self.worst_angle:
            return 0.0

        # 선형 감점
        range_size = self.worst_angle - self.optimal_angle
        deviation = angle - self.optimal_angle
        score = 10.0 * (1 - deviation / range_size)

        return max(0.0, score)

    def calculate_single_frame(self, pose_frame, ball_velocity: Optional[np.ndarray]) -> Tuple[Optional[float], float]:
        """
        단일 프레임의 정렬 각도와 점수

        Returns:
            (angle, score): 각도와 점수 튜플
        """
        body_forward = self._calculate_body_forward(pose_frame)

        if body_forward is None or ball_velocity is None:
            return None, 5.0  # 데이터 부족시 중간 점수

        body_forward_h = self._project_to_horizontal(body_forward)
        ball_velocity_h = self._project_to_horizontal(ball_velocity)

        angle = self._angle_between_vectors(body_forward_h, ball_velocity_h)
        score = self._score_alignment(angle)

        return angle, score
