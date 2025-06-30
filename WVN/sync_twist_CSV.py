#!/usr/bin/env python3
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser(
        description="current_twist.csv(많음)와 desired_twist.csv(적음)를 시간 기준으로 매칭하여, "
                    "desired 개수만큼의 current_twist만 출력"
    )
    p.add_argument('-c','--current', required=True,
                   help="Current twist CSV 경로 (e.g. 142863 rows)")
    p.add_argument('-d','--desired', required=True,
                   help="Desired twist CSV 경로 (e.g.  4470 rows)")
    p.add_argument('-o','--output', default='matched_current_twist.csv',
                   help="출력 CSV 경로 (default: %(default)s)")
    p.add_argument('-t','--tolerance', type=float, default=None,
                   help="merge_asof tolerance (초), 지정하지 않으면 제한 없음")
    args = p.parse_args()

    # 1) CSV 읽기
    curr = pd.read_csv(args.current)
    des  = pd.read_csv(args.desired)

    # 2) sec + nsec → float timestamp
    curr['time_curr'] = curr['sec'] + curr['nsec'] * 1e-9
    des ['time_des']  = des ['sec']  + des ['nsec']  * 1e-9

    # 3) 정렬
    curr = curr.sort_values('time_curr').reset_index(drop=True)
    des  = des.sort_values('time_des' ).reset_index(drop=True)

    # 4) desired(왼쪽) ↔ current(오른쪽) nearest 매칭
    merged = pd.merge_asof(
        des, curr,
        left_on='time_des',
        right_on='time_curr',
        direction='nearest',
        suffixes=('_des','_curr'),
        tolerance=args.tolerance
    )

    # 5) 로그
    print(f"▶ desired rows: {len(des)}")
    print(f"▶ matched rows: {len(merged)}")

    # 6) current 컬럼만 뽑아서 원래 이름으로 복원
    cols_curr = ['sec','nsec',
                 'vx [m/s]','vy [m/s]','vz [m/s]',
                 'wx [rad/s]','wy [rad/s]','wz [rad/s]']
    cols_curr_merged = [f"{c}_curr" for c in cols_curr]

    df_out = merged[cols_curr_merged].copy()
    df_out.columns = cols_curr

    # 7) 저장
    df_out.to_csv(args.output, index=False)
    print(f"▶ '{args.output}' 생성 완료 ({len(df_out)} rows, {len(cols_curr)} cols)")

if __name__ == '__main__':
    main()