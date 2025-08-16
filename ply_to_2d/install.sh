#!/bin/bash
# PLY to Top-View Converter 설치 스크립트

echo "PLY to Top-View Converter 설치 중..."

# Python 3 확인
if ! command -v python3 &> /dev/null; then
    echo "오류: Python 3가 설치되어 있지 않습니다."
    exit 1
fi

# pip 확인
if ! command -v pip3 &> /dev/null; then
    echo "오류: pip3가 설치되어 있지 않습니다."
    exit 1
fi

# 필요한 패키지 설치
echo "필요한 Python 패키지를 설치합니다..."
pip3 install -r requirements.txt

# 실행 권한 부여
chmod +x ply_to_topview.py
chmod +x examples/example_usage.py

echo "설치 완료!"
echo ""
echo "사용법:"
echo "  python3 ply_to_topview.py input.ply"
echo "  python3 ply_to_topview.py --help"
echo ""
echo "예제 실행:"
echo "  cd examples && python3 example_usage.py"

