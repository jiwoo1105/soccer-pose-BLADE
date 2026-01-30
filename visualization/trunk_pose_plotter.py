# visualization/trunk_pose_plotter.py
"""
상체 자세 그래프 생성 모듈
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path


class TrunkPosePlotter:
    """
    상체 자세를 그래프로 시각화하는 클래스
    """

    def __init__(self, use_korean_font: bool = True):
        """
        TrunkPosePlotter 초기화

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

    def plot_trunk_angle(self, trunk_pose_data, save_path: Optional[str] = None):
        """
        상체 각도 그래프 시각화

        Args:
            trunk_pose_data: TrunkPoseData 객체
            save_path: 그래프 저장 경로 (None이면 화면에 표시만)
        """
        # 그래프 생성
        fig, ax = plt.subplots(1, 1, figsize=(14, 7))
        fig.suptitle('상체 각도 분석 (Trunk Angle)', fontsize=16, fontweight='bold')

        # 상체 각도 그래프
        self._plot_angle_over_time(ax, trunk_pose_data)

        # 레이아웃 조정
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 저장 또는 표시
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 상체 각도 그래프 저장: {save_path}")
            plt.close()
        else:
            plt.show()

    def _plot_angle_over_time(self, ax, trunk_pose_data):
        """
        프레임별 상체 각도 변화 그래프 그리기

        Args:
            ax: matplotlib axis
            trunk_pose_data: TrunkPoseData 객체
        """
        frames = trunk_pose_data.frame_numbers
        angles = trunk_pose_data.trunk_angles

        # 각도 플롯 (메인 라인)
        ax.plot(frames, angles, 'b-', linewidth=2, label='상체 각도', alpha=0.7)

        # 축 레이블 및 제목
        ax.set_xlabel('프레임 번호', fontsize=12, fontweight='bold')
        ax.set_ylabel('상체 각도 (degrees)', fontsize=12, fontweight='bold')
        ax.set_title('프레임별 상체 각도 변화 (무릎-엉덩이-어깨)', fontsize=13, fontweight='bold', pad=10)

        # 그리드
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

        # 범례
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
