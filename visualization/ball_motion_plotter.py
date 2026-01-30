# visualization/ball_motion_plotter.py
"""
=============================================================================
공 움직임 그래프 생성 모듈
=============================================================================

이 모듈은 공의 움직임 데이터를 시각화하여 그래프로 저장합니다.

주요 기능:s
1. 공의 x, y 좌표 변화 그래프
2. 속도(velocity magnitude) 그래프
3. Peak 지점(터치 순간) 표시

활용:
>>> plotter = BallMotionPlotter()
>>> plotter.plot_motion(motion_data, save_path="output/graphs/ball_motion.png")
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from typing import Optional
from pathlib import Path


class BallMotionPlotter:
    """
    공의 움직임을 그래프로 시각화하는 클래스
    """

    def __init__(self, use_korean_font: bool = True):
        """
        BallMotionPlotter 초기화

        Args:
            use_korean_font: 한글 폰트 사용 여부
        """
        # 한글 폰트 설정
        if use_korean_font:
            self._setup_korean_font()

    def _setup_korean_font(self):
        """한글 폰트 설정 (맥OS용)"""
        try:
            # macOS의 AppleGothic 폰트 사용
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        except:
            # 폰트 설정 실패 시 기본 폰트 사용
            print("Warning: 한글 폰트 설정 실패. 기본 폰트를 사용합니다.")

    def plot_motion(self, motion_data, save_path: Optional[str] = None):
        """
        공의 속도 그래프 시각화

        Args:
            motion_data: BallMotionData 객체
            save_path: 그래프 저장 경로 (None이면 화면에 표시만)
        """
        # 1개의 플롯 생성 (속도만)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        fig.suptitle('공 속도 분석', fontsize=16, fontweight='bold')

        # 속도 크기 그래프
        self._plot_velocity(ax, motion_data)

        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 저장 또는 표시
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def _plot_velocity(self, ax, motion_data):
        """
        속도 크기 그래프 그리기

        Args:
            ax: matplotlib axis
            motion_data: BallMotionData 객체
        """
        frames = motion_data.frame_numbers
        velocities = motion_data.velocity_magnitudes

        # 속도 플롯
        ax.plot(frames, velocities, 'g-', linewidth=1.5, label='속도 크기')

        # 터치 지점 표시
        touch_frames = motion_data.touch_frames
        touch_velocities = []
        for tf in touch_frames:
            if tf in frames:
                touch_idx = np.where(frames == tf)[0][0]
                touch_velocities.append(velocities[touch_idx])
            else:
                closest_idx = np.argmin(np.abs(frames - tf))
                touch_velocities.append(velocities[closest_idx])

        ax.scatter(touch_frames, touch_velocities, color='red',
                  s=100, zorder=5, label='터치 지점 (Peak)')

        # 축 레이블 및 제목
        ax.set_xlabel('프레임 번호', fontsize=11)
        ax.set_ylabel('속도 크기 (pixels/frame)', fontsize=11)
        ax.set_title('공의 속도 변화', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    def plot_trajectory_2d(self, motion_data, save_path: Optional[str] = None):
        """
        공의 2D 궤적을 평면에 그리기 (위에서 본 시점)

        Args:
            motion_data: BallMotionData 객체
            save_path: 그래프 저장 경로
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        positions = motion_data.positions
        x = positions[:, 0]
        y = positions[:, 1]

        # 궤적 플롯 (색상 그라디언트: 시간 순서)
        scatter = ax.scatter(x, y, c=motion_data.frame_numbers,
                           cmap='viridis', s=30, alpha=0.6)
        ax.plot(x, y, 'b-', linewidth=1, alpha=0.3)

        # 터치 지점 표시
        touch_frames = motion_data.touch_frames
        touch_x = []
        touch_y = []
        frames = motion_data.frame_numbers
        for tf in touch_frames:
            if tf in frames:
                touch_idx = np.where(frames == tf)[0][0]
                touch_x.append(x[touch_idx])
                touch_y.append(y[touch_idx])

        ax.scatter(touch_x, touch_y, color='red', s=200,
                  marker='*', zorder=5, label='터치 지점')

        # 컬러바
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('프레임 번호', fontsize=11)

        # 축 레이블 및 제목
        ax.set_xlabel('X 좌표 (pixels)', fontsize=12)
        ax.set_ylabel('Y 좌표 (pixels)', fontsize=12)
        ax.set_title('공의 2D 궤적', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        # y축 반전 (이미지 좌표계는 위가 0)
        ax.invert_yaxis()

        plt.tight_layout()

        # 저장 또는 표시
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"궤적 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()
