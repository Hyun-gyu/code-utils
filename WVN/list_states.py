# list_states.py
# 목적: rosbag 파일에서 '/wild_visual_navigation_node/robot_state' 토픽으로부터 수신된
#       CustomState 메시지들의 상태 이름(state name) 출현 빈도를 집계하고 출력함.

from collections import Counter  # 상태 이름의 빈도수를 세기 위한 Counter 모듈
import rosbag 

# 분석할 rosbag 파일 경로 지정
bag = rosbag.Bag('/dev/ssd2/hyungyu/datasets/WVN/2023-09-20-09-43-57_Hiking_Utilberg-001.bag', 'r')

cnt = Counter()  # 상태 이름 빈도를 저장할 Counter 객체

# rosbag에서 '/wild_visual_navigation_node/robot_state' 토픽 메시지를 반복적으로 읽음
for _, msg, _ in bag.read_messages(topics=['/wild_visual_navigation_node/robot_state']):
    for st in msg.states:  # 각 메시지 안의 states 필드를 순회
        cnt[st.name] += 1  # 상태 이름을 기준으로 출현 횟수 집계

bag.close()  # rosbag 닫기

# 결과 출력
print("CustomState name counts:")
for name, c in cnt.items():
    print(f"  {name}: {c}")  # 상태 이름별 출현 횟수 출력

    # msg 안에 state 필드는 뭐야? 