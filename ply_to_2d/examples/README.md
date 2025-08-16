# 사용 예제

이 디렉토리는 PLY to Top-View Converter의 다양한 사용 예제를 포함합니다.

## 예제 파일

### 1. example_usage.py
프로그래밍 방식으로 툴을 사용하는 예제입니다.

```bash
cd /root/ply_to_2d/examples
python example_usage.py
```

**포함된 예제들:**
- 단일 파일 처리
- 커스텀 매개변수 설정
- 배치 처리

### 2. 명령줄 사용 예제

```bash
# 기본 사용법
python ../ply_to_topview.py sample.ply

# 지면 평면 시각화 포함
python ../ply_to_topview.py sample.ply --visualize-ground-plane

# 커스텀 해상도
python ../ply_to_topview.py sample.ply --resolutions 0.01 0.02 0.05

# 배치 처리
python ../ply_to_topview.py /path/to/ply/files/ --output batch_results/
```

## 예제 데이터

실제 PLY 파일이 필요합니다. 다음 형식을 지원합니다:

### PLY 파일 구조 예제
```
ply
format binary_little_endian 1.0
element vertex 1000000
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
[binary data...]
```

## 성능 벤치마크

### 테스트 환경
- 파일: RELLIS-3D 데이터셋 (1.3M 포인트)
- 하드웨어: 표준 데스크톱

### 처리 시간 (근사치)
| 해상도 | 처리 시간 | 이미지 크기 |
|--------|-----------|-------------|
| 0.05m/pixel | ~30초 | 99×71 |
| 0.03m/pixel | ~45초 | 165×119 |
| 0.02m/pixel | ~60초 | 247×177 |
| 0.01m/pixel | ~90초 | 493×354 |

## 출력 예제

생성되는 파일들:
```
example_results/
└── sample_file/
    ├── sample_file_topview_0p050m.png        # 저해상도 (축 포함)
    ├── sample_file_topview_0p050m_clean.png  # 저해상도 (깨끗한 버전)
    ├── sample_file_topview_0p030m.png        # 중해상도 (축 포함)
    ├── sample_file_topview_0p030m_clean.png  # 중해상도 (깨끗한 버전)
    ├── sample_file_ground_plane.png          # 지면 평면 시각화
    └── sample_file_metadata.json             # 메타데이터
```

## 커스텀 사용법

### Python 스크립트에서 직접 사용

```python
from ply_to_topview import PLYProcessor

# 프로세서 생성
processor = PLYProcessor(
    output_dir="my_results",
    visualize_ground_plane=True
)

# 단일 파일 처리
processor.process_single_ply("my_file.ply", resolutions=[0.02, 0.03])

# 또는 개별 단계 실행
points, normals, colors = processor.read_ply_binary("my_file.ply")
ground_normal, _ = processor.detect_ground_plane(points)
image, bounds, size = processor.create_topview_image(points, colors, ground_normal)
```

## 문제해결 예제

### 1. 대용량 파일 처리
```python
# 메모리 절약을 위한 서브샘플링
import numpy as np

if len(points) > 1000000:
    indices = np.random.choice(len(points), 1000000, replace=False)
    points = points[indices]
    colors = colors[indices]
```

### 2. 커스텀 지면 평면
```python
# 수동으로 지면 법선 벡터 설정
custom_normal = np.array([0, 0, 1])  # 완전히 수평인 지면
image, bounds, size = processor.create_topview_image(
    points, colors, custom_normal
)
```

### 3. 색상이 없는 PLY 파일
```python
# 색상 정보가 없는 경우 높이 기반 색칠
if colors is None or np.all(colors == 0):
    z_coords = points[:, 2]
    z_norm = (z_coords - np.min(z_coords)) / (np.max(z_coords) - np.min(z_coords))
    colors = plt.cm.viridis(z_norm)[:, :3]  # 높이 맵 색칠
```

