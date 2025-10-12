# visualization/angle_plotter.py
"""
각도 그래프 생성
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import seaborn as sns


class AnglePlotter:
    """
    각도 데이터 시각화
    """
    
    def __init__(self, figsize: tuple = (12, 6), dpi: int = 100):
        """
        Args:
            figsize: 그래프 크기
            dpi: 해상도
        """
        self.figsize = figsize
        self.dpi = dpi
        sns.set_style("whitegrid")
    
    def plot_comparison(self, 
                       angles1: np.ndarray, 
                       angles2: np.ndarray,
                       title: str = "Angle Comparison",
                       labels: List[str] = ['Video 1', 'Video 2'],
                       save_path: Optional[str] = None) -> None:
        """
        두 각도 시계열을 비교하는 그래프
        
        Args:
            angles1: 첫 번째 각도 배열
            angles2: 두 번째 각도 배열
            title: 그래프 제목
            labels: 범례 라벨
            save_path: 저장 경로 (None이면 저장 안함)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 시간축 생성 (프레임 번호)
        time1 = np.arange(len(angles1))
        time2 = np.arange(len(angles2))
        
        # 그래프 그리기
        ax.plot(time1, angles1, 'b-', label=labels[0], linewidth=2, alpha=0.7)
        ax.plot(time2, angles2, 'r-', label=labels[1], linewidth=2, alpha=0.7)
        
        # 평균선
        ax.axhline(y=np.mean(angles1), color='b', linestyle='--', 
                   alpha=0.5, label=f'{labels[0]} Mean: {np.mean(angles1):.1f}°')
        ax.axhline(y=np.mean(angles2), color='r', linestyle='--', 
                   alpha=0.5, label=f'{labels[1]} Mean: {np.mean(angles2):.1f}°')
        
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Angle (degrees)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_statistics_bars(self, 
                            comparison_df,
                            metric: str = 'Mean_Diff',
                            save_path: Optional[str] = None) -> None:
        """
        비교 통계를 막대 그래프로 표시
        
        Args:
            comparison_df: 비교 DataFrame
            metric: 표시할 메트릭 컬럼명
            save_path: 저장 경로
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        joints = comparison_df['Joint/Segment']
        values = comparison_df[metric]
        
        # 색상 (양수/음수에 따라)
        colors = ['red' if v < 0 else 'blue' for v in values]
        
        bars = ax.barh(joints, values, color=colors, alpha=0.7)
        
        # 0선
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel(f'{metric} (degrees)', fontsize=12)
        ax.set_title(f'Comparison: {metric}', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
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
        
        Args:
            angles1: 첫 번째 각도 배열
            angles2: 두 번째 각도 배열
            title: 그래프 제목
            labels: 범례 라벨
            save_path: 저장 경로
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # 히스토그램
        ax.hist(angles1, bins=20, alpha=0.5, label=labels[0], color='blue')
        ax.hist(angles2, bins=20, alpha=0.5, label=labels[1], color='red')
        
        # 평균선
        ax.axvline(np.mean(angles1), color='blue', linestyle='--', linewidth=2,
                   label=f'{labels[0]} Mean: {np.mean(angles1):.1f}°')
        ax.axvline(np.mean(angles2), color='red', linestyle='--', linewidth=2,
                   label=f'{labels[1]} Mean: {np.mean(angles2):.1f}°')
        
        ax.set_xlabel('Angle (degrees)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()