#!/usr/bin/env python3
"""
full_trav_analysis.py

Traversability Score Analysis Pipeline

이 스크립트는 ROS bag으로부터 추출된 traversability score CSV 파일을 입력으로 받아,
1) 기본 통계치(평균, 중앙값, 표준편차, 사분위 등) 계산
2) 분포 시각화(히스토그램, KDE) 저장
3) 이상치(IQR 기반) 검출 및 CSV로 저장
4) 이상치 시점의 RGB 프레임을 ROS bag에서 추출하여 이미지 파일로 저장
결과는 지정된 --outdir 하위에 정리된 폴더 구조로 저장됩니다.

Usage:
    python3 full_trav_analysis.py \
        --score path/to/traversability.csv \
        --bag   path/to/data.bag \
        --topic /camera/image_raw \
        --outdir ./results \
        --tol   0.02

Options:
    --score  (-s) : traversability CSV 파일 경로 (timestamp, traversability, ...)
    --bag    (-b) : 원본 ROS bag 파일 경로 (이미지 프레임 추출용)
    --topic  (-t) : 이미지 토픽 이름 (기본: /camera/image_raw)
    --outdir (-d) : 분석 결과를 저장할 베이스 디렉토리
    --tol       : 프레임 매칭 시 최대 허용 시간차 (초 단위, 기본 0.02s)

목적:
    - 대량의 스코어 데이터를 통계 및 시각화를 통해 요약
    - 이상치 구간의 원인 파악을 위해 해당 시점의 영상 데이터를 함께 확보

"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rosbag
from cv_bridge import CvBridge

# -----------------------------------------------------------------------------
# 1) 기본 통계치 계산 함수
# -----------------------------------------------------------------------------
def compute_stats(scores: pd.Series, out_dir: str) -> dict:
    """
    scores 시리즈의 기본 통계량(mean, median, std, min, max, quartile) 계산 및 CSV 저장
    returns: stats dict
    """
    stats = {
        'mean':   scores.mean(),
        'median': scores.median(),
        'std':    scores.std(),
        'min':    scores.min(),
        'max':    scores.max(),
        '25%':    scores.quantile(0.25),
        '75%':    scores.quantile(0.75),
    }
    # pandas DataFrame으로 변환 후 저장
    df = pd.DataFrame.from_dict(stats, orient='index', columns=['value'])
    df.to_csv(os.path.join(out_dir, 'stats.csv'), header=True)
    return stats

# -----------------------------------------------------------------------------
# 2) 분포 시각화 함수 (히스토그램, KDE)
# -----------------------------------------------------------------------------
def plot_distribution(scores: pd.Series, out_dir: str):
    """
    scores 값을 기반으로 히스토그램 및 KDE(커널 밀도 추정) 플롯을 생성하여 PNG로 저장
    """
    # 히스토그램
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=50)
    plt.title('Traversability Histogram')
    plt.xlabel('score')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'histogram.png'))
    plt.close()

    # KDE plot
    plt.figure(figsize=(6,4))
    scores.plot.kde()
    plt.title('Traversability KDE')
    plt.xlabel('score')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kde.png'))
    plt.close()

# -----------------------------------------------------------------------------
# 3) 이상치 검출 함수 (IQR 기반)
# -----------------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    IQR 방식으로 이상치 판단.
    returns: 이상치가 포함된 df subset
    """
    s = df[col]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (s < lower) | (s > upper)
    return df[mask]

# -----------------------------------------------------------------------------
# 4) 이상치 시점 RGB 프레임 추출 함수
# -----------------------------------------------------------------------------
def extract_frames(bag_path: str, out_dir: str, timestamps: np.ndarray,
                   topic: str, tol: float):
    """
    ROS bag에서 각 timestamp와 가장 근접한 이미지 msg를 추출하여 PNG 파일로 저장
    tol: 허용 시간차(초)
    """
    bridge = CvBridge()
    bag = rosbag.Bag(bag_path)
    os.makedirs(out_dir, exist_ok=True)

    # 추출할 target timestamps 정렬
    times = np.sort(timestamps)
    idx = 0
    total = len(times)

    # 이미지 메시지를 순차적으로 읽으며 매칭
    for topic_, msg, t in bag.read_messages(topics=[topic]):
        if idx >= total:
            break
        ts_target = times[idx]
        ts_msg = t.to_sec()
        # tol 범위 내라면 저장
        if abs(ts_msg - ts_target) <= tol:
            img = bridge.imgmsg_to_cv2(msg, 'bgr8')
            # 파일명에 timestamp 포함
            fname = f"{ts_target:.6f}.png"
            filepath = os.path.join(out_dir, fname)
            import cv2
            cv2.imwrite(filepath, img)
            idx += 1
        # 메시지가 너무 과거이면 skip, 미래이면 다음 timestamp로 옮김
        elif ts_msg > ts_target + tol:
            idx += 1

    bag.close()

# -----------------------------------------------------------------------------
# 메인 함수: Argument parsing 및 전체 파이프라인 호출
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Traversability 분석 + 이상치 프레임 추출 통합 스크립트"
    )
    parser.add_argument('-s', '--score',   required=True,
                        help="Traversability CSV 파일")
    parser.add_argument('-b', '--bag',     required=True,
                        help="원본 ROS bag 파일 경로")
    parser.add_argument('-t', '--topic',   default='/camera/image_raw',
                        help="RGB 이미지 토픽 (default: %(default)s)")
    parser.add_argument('-d', '--outdir',  required=True,
                        help="결과 저장 베이스 디렉토리")
    parser.add_argument('--tol', type=float, default=0.02,
                        help="프레임 매칭 허용 시간차(초), default=0.02")
    args = parser.parse_args()

    # 출력 디렉토리 구조 생성
    base_dir      = args.outdir
    analysis_dir  = os.path.join(base_dir, 'score_analysis')
    outlier_dir   = os.path.join(base_dir, 'outliers')
    frames_dir    = os.path.join(base_dir, 'outlier_frames')
    for d in (analysis_dir, outlier_dir, frames_dir):
        os.makedirs(d, exist_ok=True)

    # CSV 로드
    df = pd.read_csv(args.score)
    scores = df['traversability']

    # 1) 기본 통계치 및 CSV 저장
    stats = compute_stats(scores, analysis_dir)

    # 2) 분포 시각화 (히스토그램 + KDE)
    plot_distribution(scores, analysis_dir)

    # 3) 이상치 검출 및 저장
    outliers = detect_outliers_iqr(df, 'traversability')
    outliers.to_csv(os.path.join(outlier_dir, 'outliers.csv'), index=False)

    # 4) 이상치 시점 이미지 추출
    extract_frames(args.bag, frames_dir,
                   outliers['timestamp'].values,
                   args.topic, args.tol)

    # 5) README 요약 작성
    with open(os.path.join(base_dir, 'README.md'), 'w') as f:
        f.write("# Traversability Analysis Result\n\n")
        f.write("## 기본 통계치\n")
        for k, v in stats.items():
            f.write(f"- **{k}**: {v:.6f}\n")
        f.write(f"\n- 이상치 개수: {len(outliers)} (outliers/outliers.csv)\n")
        f.write("- 히스토그램: score_analysis/histogram.png\n")
        f.write("- KDE:           score_analysis/kde.png\n")
        f.write("- 이상치 프레임: outlier_frames/<timestamp>.png\n")

    print("✅ Analysis complete. results in", base_dir)

if __name__ == '__main__':
    main()