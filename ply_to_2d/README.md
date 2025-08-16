# PLY to Top-View Image Converter

PLY 포인트 클라우드 파일을 2D top-down RGB 이미지로 변환하는 Python 툴

## 주요 기능

- ✅ **단일 파일 & 배치 처리**: PLY 파일 하나 또는 디렉토리 전체 처리
- ✅ **향상된 지면 평면 검출**: 법선 벡터 정보 활용 + RANSAC fallback
- ✅ **극초고해상도 지원**: 최대 0.5mm/pixel (0.0005m) 이미지 생성
- ✅ **지능형 서브샘플링**: 해상도별 최적화된 포인트 수 관리
- ✅ **좌표계 자동 인식**: 다양한 PLY 좌표계 자동 적응
- ✅ **고급 후처리**: 노이즈 감소, CLAHE, 언샤프 마스킹 (옵션)
- ✅ **원본 방향 보존**: 좌표계 변환 없이 원본 방향 유지 옵션
- ✅ **적응적 블러**: 해상도에 따른 최적화된 보간
- ✅ **고품질 출력**: 깨끗한 이미지와 정보가 포함된 이미지 두 버전 제공
- ✅ **상세 메타데이터**: 처리 정보와 매개변수를 JSON으로 저장

## 설치

### 자동 설치 (권장)

```bash
chmod +x install.sh
./install.sh
```

### 수동 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 기본 사용법

```bash
# 단일 PLY 파일 처리 (고해상도 기본값: 2mm, 5mm, 10mm/pixel)
python3 ply_to_topview.py input.ply

# 디렉토리 내 모든 PLY 파일 배치 처리
python3 ply_to_topview.py /path/to/ply/directory/
```

### 고급 옵션

```bash
# 극초고해상도 (0.5mm/pixel)
python3 ply_to_topview.py input.ply --resolutions 0.0005 --use-all-points

# 모든 포인트 사용 + 고급 후처리
python3 ply_to_topview.py input.ply --use-all-points --advanced-postprocessing

# 원본 좌표계 방향 보존 (좌표계 변환 없음)
python3 ply_to_topview.py input.ply --preserve-orientation

# 지면 평면 검출 과정 시각화
python3 ply_to_topview.py input.ply --visualize-ground-plane

# 모든 옵션 조합
python3 ply_to_topview.py input.ply \
    --resolutions 0.0005 0.001 0.002 \
    --use-all-points \
    --advanced-postprocessing \
    --preserve-orientation \
    --visualize-ground-plane
```

### 명령줄 옵션

| 옵션 | 단축 | 설명 | 기본값 |
|------|------|------|--------|
| `input` | - | 입력 PLY 파일 또는 디렉토리 | 필수 |
| `--output` | `-o` | 출력 디렉토리 | `results` |
| `--resolutions` | `-r` | 해상도 리스트 (m/pixel) | `0.002 0.005 0.01` |
| `--visualize-ground-plane` | `-v` | 지면 평면 시각화 활성화 | 비활성화 |
| `--advanced-postprocessing` | `-a` | 고급 후처리 적용 | 비활성화 |
| `--use-all-points` | `-u` | 모든 포인트 사용 (서브샘플링 비활성화) | 비활성화 |
| `--preserve-orientation` | `-p` | 원본 좌표계 방향 보존 | 비활성화 |

## 출력 구조

```
results/
├── filename1/
│   ├── filename1_topview_0p0005m.png       # 0.0005m/pixel (축 포함)
│   ├── filename1_topview_0p0005m_clean.png # 0.0005m/pixel (깨끗한 버전)
│   ├── filename1_topview_0p001m.png        # 0.001m/pixel (축 포함)
│   ├── filename1_topview_0p001m_clean.png  # 0.001m/pixel (깨끗한 버전)
│   ├── filename1_topview_0p002m.png        # 0.002m/pixel (축 포함)
│   ├── filename1_topview_0p002m_clean.png  # 0.002m/pixel (깨끗한 버전)
│   ├── filename1_ground_plane.png          # 지면 평면 시각화 (옵션)
│   └── filename1_metadata.json             # 처리 메타데이터
└── filename2/
    └── ...
```

## 알고리즘 개요

### 1. 향상된 지면 평면 검출
- **1순위**: PLY 파일의 법선 벡터 정보 활용
  - 수직에 가까운 법선 벡터들 자동 감지
  - 법선 벡터 평균으로 지면 방향 추정
- **2순위**: RANSAC 알고리즘 fallback
  - Z값 하위 20% 포인트를 지면 후보로 선택
  - 2000회 반복으로 최적 평면 피팅

### 2. 지능형 좌표계 처리
- **기본 모드**: 지면 평면을 Z축과 정렬하여 표준화
- **원본 보존 모드**: 좌표계 변환 없이 원본 방향 유지
- **자동 방향 인식**: 다양한 PLY 좌표계 자동 적응

### 3. 2D 투영 및 렌더링
- X, Y 좌표를 이미지 픽셀로 매핑
- 깊이 버퍼링으로 가장 높은 점의 색상 사용
- 적응적 가우시안 블러로 빈 픽셀 보간

### 4. 지능형 서브샘플링
- **극초고해상도 (≤0.001m)**: 150만 포인트까지 사용
- **초고해상도 (≤0.002m)**: 제한 없음
- **일반 해상도 (>0.002m)**: 100만 포인트 제한
- **--use-all-points**: 모든 제한 해제

### 5. 고급 후처리 (옵션)
- **노이즈 감소**: 메디안 필터 (3×3 커널)
- **적응적 히스토그램 균등화**: CLAHE (clipLimit=2.0)
- **언샤프 마스킹**: 5mm/pixel 이하에서 선명도 향상 (σ=1.0, 강도=0.5)
- **색상 보정**: 밝기 및 채도 최적화

## 해상도 비교 및 성능

| 해상도 | 파일명 | 이미지 크기 예시 | 파일 크기 | 처리 시간 | 용도 |
|--------|--------|------------------|-----------|-----------|------|
| 0.01m/pixel | 0p01m | 566×263 | 59KB | ~15초 | 일반적인 시각화 |
| 0.005m/pixel | 0p005m | 1131×524 | 198KB | ~20초 | 상세한 분석 |
| 0.002m/pixel | 0p002m | 2828×1311 | 1.2MB | ~30초 | 고정밀 매핑 |
| 0.001m/pixel | 0p001m | 5656×2622 | 4.8MB | ~60초 | 극초고해상도 |
| **0.0005m/pixel** | **0p0005m** | **11312×5244** | **19MB** | **~120초** | **극한 정밀도** |

### 적응적 블러 강도
- **극초고해상도 (≤0.001m)**: σ=1.5 (강한 보간)
- **초고해상도 (≤0.002m)**: σ=1.2 (중간 보간)
- **일반 해상도 (>0.002m)**: σ=0.8 (기본 보간)

## 지원 PLY 형식

- **좌표**: 필수 (x, y, z)
- **법선**: 선택사항 (nx, ny, nz) - 지면 검출에 활용
- **색상**: 선택사항 (r, g, b) - 없으면 회색으로 설정
- **형식**: Binary Little Endian PLY

## 사용 시나리오

### 1. 표준 처리 (권장)
```bash
python3 ply_to_topview.py input.ply
```
- 기본 해상도 (2mm, 5mm, 10mm/pixel)
- 자동 지면 검출 및 좌표계 정렬
- 적절한 성능과 품질 균형

### 2. 극초고해상도 처리
```bash
python3 ply_to_topview.py input.ply --resolutions 0.0005 --use-all-points -a
```
- 0.5mm/pixel 해상도
- 모든 포인트 사용
- 고급 후처리 적용

### 3. 원본 방향 보존 (좌표계 문제 해결)
```bash
python3 ply_to_topview.py input.ply --preserve-orientation
```
- Meshlab/CloudCompare와 동일한 방향
- 좌표계 변환 없음
- 다양한 PLY 좌표계 대응

### 4. 디버깅 및 분석
```bash
python3 ply_to_topview.py input.ply -v -a --use-all-points
```
- 지면 평면 검출 과정 시각화
- 고급 후처리로 세부사항 강화
- 모든 포인트 정보 활용

## 문제해결

### 일반적인 문제

1. **이미지가 뒤집혀 보임**
   ```bash
   python3 ply_to_topview.py input.ply --preserve-orientation
   ```

2. **이미지가 너무 sparse함**
   ```bash
   python3 ply_to_topview.py input.ply --use-all-points
   ```

3. **지면 평면 검출 실패**
   - PLY 파일에 법선 벡터가 있는지 확인
   - `--visualize-ground-plane` 옵션으로 검출 과정 확인

4. **메모리 부족**
   - 더 낮은 해상도 사용 (기본 서브샘플링 활용)
   - `--use-all-points` 옵션 제거

### 성능 최적화 팁

- **빠른 처리**: 기본 옵션 사용
- **고품질**: `--advanced-postprocessing` 추가
- **극한 품질**: `--use-all-points --advanced-postprocessing` 조합
- **배치 처리**: 디렉토리 입력으로 여러 파일 자동 처리

## 기술적 세부사항

### 지면 평면 검출 알고리즘
1. **법선 벡터 방법** (우선순위)
   - 법선 벡터 길이 > 0.1 확인
   - Z축과의 각도 계산 (cos > 0.5)
   - 수직 법선들의 평균으로 지면 방향 추정
   
2. **RANSAC 방법** (fallback)
   - 반복 횟수: 2000회
   - 임계값: 0.05m
   - 최소 인라이어: 1000개

### 고급 후처리 파라미터
- **메디안 필터**: 3×3 커널
- **CLAHE**: clipLimit=2.0, tileGridSize=8×8
- **언샤프 마스킹**: σ=1.0, 강도=0.5 (5mm/pixel 이하)

## 예제 파일

`examples/` 디렉토리에서 샘플 데이터와 실행 예제를 확인하세요.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다.

## 버전 히스토리

- **v4.0.0** (2024년): 최종 통합 버전
  - 법선 벡터 정보 활용한 향상된 지면 검출
  - 극초고해상도 지원 (0.5mm/pixel)
  - 지능형 서브샘플링 시스템
  - 원본 좌표계 방향 보존 옵션
  - 적응적 블러 및 고급 후처리
  - 사용성 및 안정성 대폭 개선

- **v3.0.0** (2024년): 통합 버전
  - ultra_high_res.py 기능을 메인 스크립트에 통합
  - 옵션으로 선택 가능한 고급 후처리
  - 기본 해상도를 고해상도로 변경

- **v2.0.0** (2024-08-16): 초고해상도 지원
  - 타일링 방식으로 1mm/pixel 해상도 지원
  - 슈퍼 해상도 업스케일링
  - 고급 이미지 후처리 기술

- **v1.0.0** (2024-08-16): 초기 릴리스
  - 기본 PLY to 2D 변환 기능
  - 자동 지면 평면 검출
  - 다중 해상도 지원