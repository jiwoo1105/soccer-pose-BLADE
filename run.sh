#!/bin/bash
# 축구 드리블 분석 시스템 실행 스크립트

cd "$(dirname "$0")"

# 가상환경 활성화
source venv/bin/activate

# 실행
python main.py "$@"
