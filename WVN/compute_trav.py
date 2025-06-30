#!/usr/bin/env python3
import os
import torch
import pandas as pd
from wild_visual_navigation.supervision_generator import SupervisionGenerator

def compute_and_save_traversability(
    current_csv: str,
    desired_csv: str,
    out_csv: str,
):
    # 1) CSV 로드
    df_curr = pd.read_csv(current_csv)
    df_des  = pd.read_csv(desired_csv)

    # 2) 컬럼명에서 단위 제거: "vx [m/s]" -> "vx", "wy [rad/s]" -> "wy"
    def strip_unit(col: str) -> str:
        # 공백으로 분리한 첫 토큰을 변수명으로 사용
        return col.split()[0]
    df_curr.rename(columns=strip_unit, inplace=True)
    df_des .rename(columns=strip_unit, inplace=True)

    # 3) 길이·타임스탬프 일치 확인
    assert len(df_curr) == len(df_des), "current/desired CSV 길이 불일치"
    # assert (df_curr['sec'].equals(df_des['sec']) and
    #         df_curr['nsec'].equals(df_des['nsec'])), "타임스탬프(sec/nsec)가 일치하지 않음"

    # 4) SupervisionGenerator 초기화
    sg = SupervisionGenerator(
        device="cpu",
        kf_process_cov=0.1,
        kf_meas_cov=1000.0,
        kf_outlier_rejection="huber",
        kf_outlier_rejection_delta=0.5,
        sigmoid_slope=30.0,
        sigmoid_cutoff=0.2,
        untraversable_thr=0.05,
        time_horizon=0.05,
        graph_max_length=1.0,
    )

    rows = []
    # 5) itertuples 순회
    for c_row, d_row in zip(df_curr.itertuples(index=False), df_des.itertuples(index=False)):
        ts = c_row.sec + c_row.nsec * 1e-9

        curr_vel = torch.tensor([
            c_row.vx, c_row.vy, c_row.vz,
            c_row.wx, c_row.wy, c_row.wz
        ], dtype=torch.float32)
        des_vel  = torch.tensor([
            d_row.vx, d_row.vy, d_row.vz,
            d_row.wx, d_row.wy, d_row.wz
        ], dtype=torch.float32)

        trav, trav_var, is_untrav = sg.update_velocity_tracking(
            current_velocity=curr_vel,
            desired_velocity=des_vel,
            max_velocity=1.0,
            velocities=["vx","vy","vz","wx","wy","wz"],
        )

        rows.append({
            "timestamp":       ts,
            "traversability":  trav.item(),
            "variance":        trav_var.item(),
            "is_untraversable": int(is_untrav),
        })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved {len(out_df)} entries to {out_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="twist CSV → traversability CSV")
    p.add_argument("--curr", required=True, help="current twist CSV 경로")
    p.add_argument("--des",  required=True, help="desired twist CSV 경로")
    p.add_argument("--out",  required=True, help="출력할 traversability CSV 경로")
    args = p.parse_args()

    compute_and_save_traversability(
        current_csv=args.curr,
        desired_csv=args.des,
        out_csv=args.out,
    )