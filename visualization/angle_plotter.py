# visualization/angle_plotter.py
"""
=============================================================================
각도 그래프 생성
=============================================================================

이 모듈은 Matplotlib을 사용하여 각도 데이터를 시각화합니다.

주요 그래프:
1. 시계열 비교: 시간에 따른 각도 변화 비교
2. 분포 히스토그램: 각도 사용 빈도 비교
3. 막대 그래프: 통계량 비교

라이브러리:
- matplotlib: 그래프 그리기
- seaborn: 스타일링

활용:
>>> plotter = AnglePlotter()
>>> plotter.plot_comparison(angles1, angles2, labels=["선수A", "선수B"])
>>> plotter.plot_distribution(angles1, angles2)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import seaborn as sns


class AnglePlotter:
    """
    각도 데이터 시각화 클래스

    역할:
    1. 시계열 그래프: 프레임별 각도 변화 추적
    2. 히스토그램: 각도 분포 비교
    3. 막대 그래프: 통계량 시각화

    matplotlib 사용법:
    - plt.figure(): 그래프 생성
    - ax.plot(): 선 그래프
    - ax.hist(): 히스토그램
    - plt.savefig(): 파일로 저장
    """

    def __init__(self, figsize: tuple = (12, 6), dpi: int = 100):
        """
        AnglePlotter 초기화

        Args:
            figsize: 그래프 크기 (width, height) in inches
                    예: (12, 6) = 12인치 x 6인치
            dpi: 해상도 (dots per inch)
                - 100: 일반적인 화면 표시용
                - 300: 인쇄용 고해상도

        사용 예시:
            >>> # 큰 그래프, 고해상도
            >>> plotter = AnglePlotter(figsize=(16, 8), dpi=150)
        """
        self.figsize = figsize
        self.dpi = dpi

        # Seaborn 스타일 설정 (깔끔한 그리드 배경)
        sns.set_style("whitegrid")

    def plot_comparison(self,
                       angles1: np.ndarray,
                       angles2: np.ndarray,
                       title: str = "Angle Comparison",
                       labels: List[str] = ['Video 1', 'Video 2'],
                       save_path: Optional[str] = None) -> None:
        """
        두 각도 시계열을 비교하는 그래프

        그래프 구성:
        - X축: 프레임 번호 (시간)
        - Y축: 각도 (degrees)
        - 파란 선: 첫 번째 비디오
        - 빨간 선: 두 번째 비디오
        - 점선: 각 비디오의 평균값

        Args:
            angles1: 첫 번째 각도 배열
                    예: [152.3, 148.7, 149.1, ...]
            angles2: 두 번째 각도 배열
            title: 그래프 제목
            labels: 범례 라벨 [label1, label2]
            save_path: 저장 경로 (None이면 화면에만 표시)
                      예: "output/plots/knee_comparison.png"

        활용:
            >>> plotter = AnglePlotter()
            >>> plotter.plot_comparison(
            >>>     motion1_data['left_knee_angles'],
            >>>     motion2_data['left_knee_angles'],
            >>>     title="Left Knee Angle Comparison",
            >>>     labels=["Soccer 1", "Soccer 2"],
            >>>     save_path="output/plots/knee.png"
            >>> )

        그래프 해석:
            - 두 선이 가까우면: 유사한 패턴
            - 평균선 차이: 전체적인 자세 차이
            - 진폭 차이: 다이나믹함의 차이
        """
        # STEP 1: 그래프 생성
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # STEP 2: 시간축 생성 (프레임 번호)
        time1 = np.arange(len(angles1))  # [0, 1, 2, ..., n-1]
        time2 = np.arange(len(angles2))

        # STEP 3: 시계열 그래프 그리기
        # 'b-': blue solid line
        # 'r-': red solid line
        # alpha: 투명도 (0=투명, 1=불투명)
        ax.plot(time1, angles1, 'b-', label=labels[0], linewidth=2, alpha=0.7)
        ax.plot(time2, angles2, 'r-', label=labels[1], linewidth=2, alpha=0.7)

        # STEP 4: 평균선 그리기
        # axhline: 수평선 (horizontal line)
        ax.axhline(y=np.mean(angles1), color='b', linestyle='--',
                   alpha=0.5, label=f'{labels[0]} Mean: {np.mean(angles1):.1f}°')
        ax.axhline(y=np.mean(angles2), color='r', linestyle='--',
                   alpha=0.5, label=f'{labels[1]} Mean: {np.mean(angles2):.1f}°')

        # STEP 5: 축 라벨 및 제목
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # STEP 6: 범례 및 그리드
        ax.legend(loc='best', fontsize=10)  # 최적 위치에 범례 표시
        ax.grid(True, alpha=0.3)  # 반투명 그리드

        # 여백 조정
        plt.tight_layout()

        # STEP 7: 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()  # 화면에 표시

        # 메모리 정리
        plt.close()

    def plot_statistics_bars(self,
                            comparison_df,
                            metric: str = 'Mean_Diff',
                            save_path: Optional[str] = None) -> None:
        """
        비교 통계를 막대 그래프로 표시

        용도:
        - 여러 관절/분절의 차이를 한눈에 비교
        - 양수/음수를 색상으로 구분

        Args:
            comparison_df: MotionComparison.compare_characteristics()의 결과
                          DataFrame with columns: 'Joint/Segment', 'Mean_Diff', etc.
            metric: 표시할 메트릭 컬럼명
                   - 'Mean_Diff': 평균값 차이
                   - 'ROM_Diff': 가동 범위 차이
            save_path: 저장 경로

        그래프 구성:
        - X축: 차이값 (degrees)
        - Y축: 관절/분절 이름
        - 파란색 막대: Video1이 더 큼 (양수)
        - 빨간색 막대: Video2가 더 큼 (음수)

        예시:
            >>> table = comparator.compare_characteristics()
            >>> plotter.plot_statistics_bars(
            >>>     table,
            >>>     metric='Mean_Diff',
            >>>     save_path="output/plots/diff_bars.png"
            >>> )
        """
        # STEP 1: 그래프 생성
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)

        # STEP 2: 데이터 추출
        joints = comparison_df['Joint/Segment']
        values = comparison_df[metric]

        # STEP 3: 색상 결정 (양수/음수에 따라)
        # 양수 (Video1 큼) = 파랑
        # 음수 (Video2 큼) = 빨강
        colors = ['red' if v < 0 else 'blue' for v in values]

        # STEP 4: 가로 막대 그래프
        # barh: horizontal bar chart
        bars = ax.barh(joints, values, color=colors, alpha=0.7)

        # STEP 5: 0선 표시 (기준선)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

        # STEP 6: 축 라벨 및 제목
        ax.set_xlabel(f'{metric} (degrees)', fontsize=12)
        ax.set_title(f'Comparison: {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)  # X축에만 그리드

        plt.tight_layout()

        # STEP 7: 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_distribution(self,
                         angles1: np.ndarray,
                         angles2: np.ndarray,
                         title: str = "Angle Distribution",
                         labels: List[str] = ['Video 1', 'Video 2'],
                         save_path: Optional[str] = None) -> None:
        """
        각도 분포 히스토그램

        용도:
        - 각 비디오가 어떤 각도를 얼마나 자주 사용하는지 비교
        - 분포의 모양으로 스타일 파악

        그래프 구성:
        - X축: 각도 (degrees)
        - Y축: 빈도 (frequency)
        - 파란색 막대: Video 1
        - 빨간색 막대: Video 2
        - 점선: 평균값

        Args:
            angles1: 첫 번째 각도 배열
            angles2: 두 번째 각도 배열
            title: 그래프 제목
            labels: 범례 라벨
            save_path: 저장 경로

        히스토그램 해석:
            - 분포가 넓으면: 다양한 각도 사용 (다이나믹)
            - 분포가 좁으면: 일정한 각도 유지 (일관성)
            - 평균 위치: 전체적인 자세

        예시:
            >>> plotter.plot_distribution(
            >>>     motion1_data['left_knee_angles'],
            >>>     motion2_data['left_knee_angles'],
            >>>     title="Knee Angle Distribution",
            >>>     save_path="output/plots/knee_dist.png"
            >>> )

        시각적 비교:
            분포 모양           의미
            ────────          ─────
            넓고 평평함        다이나믹, 다양한 동작
            좁고 높음          일관성, 일정한 동작
            오른쪽으로 치우침  높은 자세 (다리 펴짐)
            왼쪽으로 치우침    낮은 자세 (다리 구부림)
        """
        # STEP 1: 그래프 생성
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # STEP 2: 히스토그램 그리기
        # bins: 막대 개수 (구간 개수)
        # alpha: 투명도 (겹쳐 보이도록)
        ax.hist(angles1, bins=20, alpha=0.5, label=labels[0], color='blue')
        ax.hist(angles2, bins=20, alpha=0.5, label=labels[1], color='red')

        # STEP 3: 평균선 표시
        # axvline: 수직선 (vertical line)
        ax.axvline(np.mean(angles1), color='blue', linestyle='--', linewidth=2,
                   label=f'{labels[0]} Mean: {np.mean(angles1):.1f}°')
        ax.axvline(np.mean(angles2), color='red', linestyle='--', linewidth=2,
                   label=f'{labels[1]} Mean: {np.mean(angles2):.1f}°')

        # STEP 4: 축 라벨 및 제목
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # STEP 5: 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

        plt.close()
