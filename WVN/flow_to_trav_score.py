#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
from scipy.special import expit

def compute_metrics(flow: np.ndarray, desired_mag: float, alpha: float, threshold: float):
    """
    flow: H×W×2 optical flow
    desired_mag: 원하는 flow magnitude (pixels/frame)
    alpha: sigmoid 스케일 파라미터
    threshold: is_untraversable 판단 임계치
    """
    mag = np.linalg.norm(flow, axis=2)        # H×W
    mean_mag = mag.mean()                     # measured flow magnitude

    # MSE vs desired
    mse = (mean_mag - desired_mag)**2

    # sigmoid 변환 → traversability score
    score = expit(-alpha * mse)

    var = mag.var()
    is_un = int(score < threshold)
    return score, var, is_un

def main():
    p = argparse.ArgumentParser(
        description="Optical flow npy + desired_twist.csv → traversability_score.csv"
    )
    p.add_argument('-i','--indir',     required=True,
                   help="Optical flow .npy 파일들이 있는 디렉토리")
    p.add_argument('-d','--desired',   required=True,
                   help="desired_twist.csv 경로 (sec,nsec,vx [m/s],vy [m/s],...)")
    p.add_argument('-o','--outcsv',    default='traversability_score.csv',
                   help="저장할 CSV 파일 경로")
    p.add_argument('--alpha',  type=float, default=10.0,
                   help="sigmoid 스케일 파라미터")
    p.add_argument('--threshold', type=float, default=0.5,
                   help="is_untraversable 판단 임계치")
    p.add_argument('--pix2m',  type=float, default=100.0,
                   help="pixel ↔ meter 변환 계수 (pixels per meter)")
    p.add_argument('--dt',     type=float, default=1/30,
                   help="프레임 간격 (초/frame), 기본 30FPS")
    args = p.parse_args()

    # 1) desired_twist.csv 읽어서 timestamp 열 생성
    df_des = pd.read_csv(args.desired)
    df_des['timestamp'] = df_des['sec'] + df_des['nsec'] * 1e-9
    df_des = df_des.sort_values('timestamp').reset_index(drop=True)

    # 2) npy 파일 목록
    files = sorted(f for f in os.listdir(args.indir) if f.endswith('.npy'))

    rows = []
    for fname in files:
        ts = float(os.path.splitext(fname)[0])  # flow timestamp
        flow = np.load(os.path.join(args.indir, fname))  # H×W×2

        # 3) 가장 가까운 desired로 매핑
        idx = df_des['timestamp'].sub(ts).abs().idxmin()
        vx, vy = df_des.loc[idx, ['vx [m/s]','vy [m/s]']].values

        # 4) desired flow magnitude 계산 (pixels/frame)
        desired_speed = np.linalg.norm([vx, vy])  # m/s
        desired_flow_mag = desired_speed * args.pix2m * args.dt

        # 5) traversability metrics
        score, var, is_un = compute_metrics(
            flow, desired_flow_mag, args.alpha, args.threshold
        )

        rows.append({
            'timestamp':        ts,
            'traversability':   score,
            'variance':         var,
            'is_untraversable': is_un
        })

    # 6) 저장
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.outcsv, index=False,
                  columns=['timestamp','traversability','variance','is_untraversable'])
    print(f"▶ Saved {len(out_df)} entries to {args.outcsv}")

if __name__ == '__main__':
    main()