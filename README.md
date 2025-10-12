<div align="center">

# ⚽ Soccer Motion Analysis

**AI 기반 축구 드리블 모션 분석 및 비교 시스템**

</div>

---

## ✨ 주요 기능

🎥 **3D 포즈 추출** - MediaPipe를 활용한 실시간 포즈 추출
📊 **스마트 비교** - 프레임별이 아닌 특성 기반 지능형 비교
📈 **시각화** - 직관적인 그래프와 리포트 자동 생성
🔍 **상세 분석** - 관절·분절 각도, 가동범위, 드리블 스타일 분류

---

## 🚀 빠른 시작

```bash
# 패키지 설치
pip install -r requirements.txt

# 영상 준비 (input 폴더에 soccer1.mp4, soccer2.mp4 추가)

# 실행
python main.py
```

---

## 📊 분석 항목

| 항목 | 측정 내용 |
|------|-----------|
| **관절 각도** | 무릎, 고관절, 발목 |
| **분절 각도** | 몸통, 허벅지, 정강이, 발 |
| **스타일 분류** | Low/High stance, 공격적/안정적 |
| **가동 범위** | ROM, 표준편차, 백분위수 |

---

## 📁 출력 결과

```
output/
  ├── data/comparison_table.csv        # 📄 비교 데이터
  ├── reports/comparison_report.txt    # 📝 상세 리포트
  └── plots/                           # 📈 그래프
      ├── *_comparison.png
      └── *_distribution.png
```

---

## 🛠️ 기술 스택

- **Computer Vision**: MediaPipe, OpenCV
- **Data Analysis**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn

---

## 📚 참고

- Dribbling determinants in sub-elite youth soccer players (2015)
- Biomechanical characteristics for identifying cutting direction (2021)