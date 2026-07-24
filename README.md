# AI 화살표 랠리 · 미로찾기 해커톤

TurtleBot3 Waffle Pi가 미로 벽의 화살표를 YOLO 분류로 읽고 좌/우 회전을 결정해 완주하는 프로젝트.

## 시스템 구성

```
[로봇 Pi]  robot_driver.py          주행·회전·안전 (scan/odom 을 로봇에서 소비 — WiFi 무관)
[노트북]   arrow_classifier_node.py YOLO11n-cls 2클래스(left/right) 분류 → /arrow_dir 발행
[통신]     /arrow_dir (std_msgs/String) 단방향 — WiFi 가 끊겨도 로봇은 벽 앞 대기가 최악
```

주행 제어와 추론을 분리한 이유: 실측에서 노트북-로봇 WiFi가 2.5초씩 통째로 멈추는 구간이
있었고, 단일 노드(노트북 실행)로는 워치독이 로봇을 계속 세웠다. `maze_racer.py` 는 분리 전
단일 노드 버전(백업).

## 실행

```bash
# 1) 로봇 (ssh 접속 후)
python3 ~/robot_driver.py            # 기본 = 최대 속도(0.26 m/s), --safe 로 감속

# 2) 노트북
source ~/venv/ros/bin/activate
python3 arrow_classifier_node.py
```

로봇은 복도와 평행하게(±45° 이내) 놓고 시작한다 — 첫 odom yaw 가 90° 격자의 기준이 된다.

## 핵심 알고리즘

- **격자 방위 고정**: 현재 yaw 에서 가장 가까운 90° 격자 방향으로 P 보정. 하드웨어
  좌편향을 상쇄하고, 회전 목표도 격자 기준이라 틀어짐이 회전마다 리셋된다.
- **벽 중앙 유지 PD**: 좌우 섹터(70–110°/250–290°) 거리 차. 전방 0.7 m 부터 페이드아웃,
  회전 직후 1.5초는 비활성(교차로 비대칭 오독 방지, 벽 25 cm 이내 밀어내기만 유지).
- **화살표 3중 게이트**: conf ≥ 0.70 + 3연속 확정(노트북) / 전방 1.5 m 이내 + 회전 직후
  1.5초 무시(로봇) / 회전 방향이 LiDAR 로 실제 뚫려 있는지 검증 후 회전.
- **피드백 회전**: odom yaw 기반 90° 회전, 오차 비례 감속, 타임아웃 안전장치.

## 주요 파라미터 (robot_driver.py 상단)

| 이름 | 값 | 의미 |
|---|---|---|
| TURN_DIST | 0.35 | 화살표 벽 앞 회전 지점 (m) |
| READ_STOP | 0.45 | 미확정 시 정지 후 분류 대기 지점 |
| SIDE_OPEN | 0.45 | 회전 전 그쪽 개방 검증 문턱 |
| KP_YAW | 1.2 | 격자 방위 보정 게인 (좌편향 보정 노브) |
| ARROW_HOLDOFF | 1.5 | 회전 후 화살표/중앙유지 홀드오프 (s) |

## 학습

```bash
python3 train_arrow_classifier.py   # arrow_dataset/(left,right) → arrow_classifier.pt
```

fliplr=0 필수(좌우 반전 증강은 라벨을 뒤집는다). 데이터셋은 인물 포함 사진이라 리포에서 제외.
