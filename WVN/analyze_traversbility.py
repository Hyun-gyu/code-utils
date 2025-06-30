#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def detect_outliers_iqr(series):
    """IQR 방식으로 이상치 인덱스 반환"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]

def main():
    p = argparse.ArgumentParser(
        description="Traversability CSV를 분석하여 통계치, 히스토그램, 이상치 결과를 저장"
    )
    p.add_argument('-i','--input',    required=True,
                   help="분석할 traversability CSV 경로")
    p.add_argument('-d','--dest',     required=True,
                   help="분석 결과를 저장할 베이스 디렉토리")
    args = p.parse_args()

    # 1) 결과 디렉토리 설정
    out_base = os.path.join(args.dest, 'score_analysis')
    os.makedirs(out_base, exist_ok=True)

    # 2) CSV 로드
    df = pd.read_csv(args.input)
    scores = df['traversability']

    # 3) 기본 통계치 계산
    stats = {
        'mean':     scores.mean(),
        'median':   scores.median(),
        'std':      scores.std(),
        'min':      scores.min(),
        'max':      scores.max(),
        '25%':      scores.quantile(0.25),
        '75%':      scores.quantile(0.75),
    }
    stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['value'])
    stats_df.to_csv(os.path.join(out_base, 'traversability_stats.csv'), header=True)

    # 4) 히스토그램/분포 시각화 저장
    plt.figure(figsize=(6,4))
    plt.hist(scores, bins=50)
    plt.title('Traversability Distribution')
    plt.xlabel('score')
    plt.ylabel('count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, 'traversability_histogram.png'))
    plt.close()

    # 5) KDE (커널 밀도) 저장
    plt.figure(figsize=(6,4))
    scores.plot.kde()
    plt.title('Traversability KDE')
    plt.xlabel('score')
    plt.tight_layout()
    plt.savefig(os.path.join(out_base, 'traversability_kde.png'))
    plt.close()

    # 6) 이상치 탐지 (IQR)
    outliers = detect_outliers_iqr(scores)
    outliers_df = df.loc[outliers.index, ['timestamp','traversability','variance','is_untraversable']]
    outliers_df.to_csv(os.path.join(out_base, 'traversability_outliers.csv'), index=False)

    # 7) 결과 요약 텍스트
    with open(os.path.join(out_base, 'README.txt'), 'w') as f:
        f.write("=== Traversability Score Analysis ===\n\n")
        f.write("기본 통계치:\n")
        for k,v in stats.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write(f"\n이상치 개수: {len(outliers_df)}\n")
        f.write("자세한 내용은 'traversability_outliers.csv' 참조\n")
        f.write("히스토그램: traversability_histogram.png\n")
        f.write("KDE 플롯:   traversability_kde.png\n")

if __name__ == '__main__':
    main()