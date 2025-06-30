#!/usr/bin/env python3
"""
rosbag에서 /wild_visual_navigation_node/robot_state 메시지를 읽어
  • current twist (geometry_msgs/TwistStamped msg.twist)
  • desired twist (msg.states 중 name=="command_twist")

두 가지를 각각 CSV로 저장하는 스크립트.

Usage:
    ./twist_to_csv.py --bag your_bag.bag --outdir ./csv_output

Outputs:
    ./csv_output/current_twist.csv
    ./csv_output/desired_twist.csv
"""

import os
import csv
import argparse
import rosbag
from wild_visual_navigation_msgs.msg import RobotState, CustomState
from geometry_msgs.msg import TwistStamped

def extract_robot_state_twists(bag_path: str, outdir: str):
    # 출력 파일 경로
    curr_csv = os.path.join(outdir, 'current_twist.csv')
    des_csv  = os.path.join(outdir, 'desired_twist.csv')

    # 파일 오픈
    f_curr = open(curr_csv, 'w', newline='')
    f_des  = open(des_csv,  'w', newline='')
    w_curr = csv.writer(f_curr)
    w_des  = csv.writer(f_des)

    # 헤더
    header = ['sec','nsec',
              'vx [m/s]','vy [m/s]','vz [m/s]',
              'wx [rad/s]','wy [rad/s]','wz [rad/s]']
    w_curr.writerow(header)
    w_des.writerow(header)

    count_curr = 0
    count_des  = 0

    with rosbag.Bag(bag_path, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=['/wild_visual_navigation_node/robot_state']):
            # msg: wild_visual_navigation_msgs/RobotState
            # --- current twist ---
            ts = msg.twist.header.stamp
            lx = msg.twist.twist.linear.x
            ly = msg.twist.twist.linear.y
            lz = msg.twist.twist.linear.z
            wx = msg.twist.twist.angular.x
            wy = msg.twist.twist.angular.y
            wz = msg.twist.twist.angular.z
            w_curr.writerow([ts.secs, ts.nsecs, lx, ly, lz, wx, wy, wz])
            count_curr += 1

            # --- desired twist: states 중 name=="command_twist" 찾기 ---
            for st in msg.states:  # CustomState[]
                if st.name == 'command_twist':
                    # CustomState.data: float[] {vx,vy,vz,wx,wy,wz}
                    sec, nsec = ts.secs, ts.nsecs
                    vx, vy, vz, awx, awy, awz = st.data
                    w_des.writerow([sec, nsec, vx, vy, vz, awx, awy, awz])
                    count_des += 1
                    break

    f_curr.close()
    f_des.close()

    print(f"[+] current_twist.csv  생성: {count_curr} 줄")
    print(f"[+] desired_twist.csv 생성: {count_des} 줄")
    print("\n미리보기 (각 파일 상위 3줄 / 하위 3줄):\n")
    for fn in [curr_csv, des_csv]:
        print("===", os.path.basename(fn), "===")
        with open(fn) as f:
            lines = f.readlines()
        preview = lines[1:4] + ["...\n"] + lines[-3:]
        for l in preview:
            print("   ", l.strip())
        print()

def main():
    p = argparse.ArgumentParser(
        description="rosbag 에서 RobotState 메시지의 current/desired twist를 CSV로 추출"
    )
    p.add_argument('--bag', '-b', required=True, help="입력 rosbag 파일 경로")
    p.add_argument('--outdir', '-o', default='.',    help="출력 디렉토리")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    extract_robot_state_twists(args.bag, args.outdir)

if __name__ == '__main__':
    main()