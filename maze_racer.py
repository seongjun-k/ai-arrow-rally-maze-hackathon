import math
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from ultralytics import YOLO

UNSTAMPED = "--unstamped" in sys.argv
MSG = Twist if UNSTAMPED else TwistStamped

SAFE = {'MAX_SPEED': 0.20, 'TURN_SPEED': 1.5}
RACE = {'MAX_SPEED': 0.22, 'TURN_SPEED': 1.8}   # Waffle Pi 최대 각속도 ~1.82 rad/s
PRESET = RACE if "--race" in sys.argv else SAFE
MAX_SPEED = PRESET['MAX_SPEED']
TURN_SPEED = PRESET['TURN_SPEED']

WALL_STOP = 0.15   # 화살표 미확정 시 최종 정지선 (TURN_DIST 보다 작아야 회전 지점까지 갈 수 있음)
SLOW_DIST = 1.0       # 여기부터 선형 감속
READ_DIST = 1.5       # 여기부터 화살표 분류 시작
INFER_PERIOD = 0.5    # YOLO 추론 최소 간격(초) — 추론이 executor 를 막아 scan 이 굶는 것 방지
KP_CENTER = 0.7       # 좌우 벽 거리 차(m) -> angular.z(rad/s) 비례 게인
KD_CENTER = 0.3       # 거리 차 변화율 감쇠 게인 (지그재그 억제)
KH_HEAD = 2.4         # 절대 헤딩 유지 게인 — 벽 없는 구간 직진 유지 (odom 전역 드리프트 시 하향)
SIDE_WALL = 0.8       # 이 거리 안이면 '벽 있음' — 중앙정렬 분기 기준 (통로 폭에 맞춰 튜닝)
SIDE_TARGET = 0.5     # 한쪽 벽만 있을 때 유지할 벽과의 목표 거리
SIDE_OPEN = 0.9       # 회전 방향이 실제로 뚫려 있다고 볼 최소 측면 거리 (막힌 벽은 ~0.5m 로 읽힘)
WAIT_ARROW_TIMEOUT = 5.0  # STOP-wait-arrow 가 이보다 길면 열린 방향으로 fallback
CONF_MIN = 0.70       # 확신도 기준
NEED = 3              # 같은 방향 연속 프레임 수
TURN_DELTA = {'left': math.pi / 2, 'right': -math.pi / 2, 'backward': math.pi}
TURN_DIST = 0.25  # 화살표 확정 상태에서 전방(화살표 벽)이 이 거리면 바로 90도 회전
READ_STOP = 0.45  # 미확정 시 여기서 멈춰 분류 (더 가면 화살표가 화면을 벗어남)
TURN_DONE = 0.05  # 목표 각도 오차(rad)가 이 이내면 회전 종료


def norm_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def valid(r):
    return r >= 0.05 and math.isfinite(r)


def sector_min(scan, lo_deg, hi_deg):
    # angle_min/angle_increment 로 인덱스를 직접 계산 (하드코딩 인덱스 금지)
    n = len(scan.ranges)
    inc = scan.angle_increment

    def idx(deg):
        return int(round((math.radians(deg) - scan.angle_min) / inc)) % n

    lo, hi = idx(lo_deg), idx(hi_deg)
    idxs = range(lo, hi + 1) if lo <= hi else list(range(lo, n)) + list(range(0, hi + 1))
    vals = [scan.ranges[i] for i in idxs if valid(scan.ranges[i])]
    return min(vals) if vals else float('inf')


def classify(model, frame, last, same, conf_min, need):
    h, w = frame.shape[:2]              # 화면 가운데 70%만 잘라 봐요
    s = int(min(h, w) * 0.7)
    crop = frame[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
    r = model.predict(crop, verbose=False)[0]
    name = model.names[r.probs.top1]
    if float(r.probs.top1conf) < conf_min:         # 확신도 낮은 프레임(모션블러)은
        return None, last, same                    # 무시하고 진행 상황 유지
    same = same + 1 if name == last else 1         # 같은 방향 연속 세기
    confirmed = name if same >= need else None     # need 번 연속이어야 확정
    return confirmed, name, same


class MazeRacer(Node):
    def __init__(self):
        super().__init__("maze_racer")
        self.model = YOLO("arrow_classifier.pt")
        # 워밍업: 첫 추론이 ~2.2초 걸려 executor 를 막음(실측) — 주행 전에 미리 1회 실행
        self.model.predict(np.zeros((224, 224, 3), np.uint8), verbose=False)
        self.pub = self.create_publisher(MSG, "/cmd_vel", 10)
        self.create_subscription(CompressedImage, "/camera/image_raw/compressed",
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_timer(0.1, self.watchdog)
        self.create_timer(0.05, self.turn_tick)   # 회전 제어 전용 (20Hz)
        now = self.get_clock().now().nanoseconds
        self.last_cam_ns = now
        self.last_scan_ns = now
        self.state = 'DRIVE'
        self.front = self.left = self.right = float('inf')
        self.yaw = None                # /odom 수신 전엔 회전 불가
        self.head = None               # 목표 절대 헤딩 — 첫 /odom 에서 초기화, 회전마다 ±90/180 누적
        self.wait_since_ns = None      # STOP-wait-arrow 진입 시각 (fallback 타이머)
        self.turn_target = 0.0
        self.turn_deadline_ns = 0      # 피드백 회전 안전 타임아웃
        self.label, self.same = '', 0
        self.confirmed = None
        self.prev_err = 0.0
        self.last_infer_ns = 0

    def send(self, lin, ang):
        msg = MSG()
        twist = msg if UNSTAMPED else msg.twist
        if not UNSTAMPED:
            msg.header.stamp = self.get_clock().now().to_msg()
        twist.linear.x = float(lin)
        twist.angular.z = float(ang)
        self.pub.publish(msg)

    def log(self, state, lin, ang):
        self.get_logger().info(
            f"state={state} front={self.front:.2f} "
            f"arrow={self.label or 'none'}({self.same}/{NEED}) "
            f"lin={lin:.2f} ang={ang:.2f}")

    def on_odom(self, msg):            # 구독: /odom (쿼터니언 -> yaw)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.head is None:
            self.head = self.yaw       # 출발 방향을 기준 헤딩으로

    def on_scan(self, msg):            # 구독: /scan
        self.last_scan_ns = self.get_clock().now().nanoseconds
        self.front = sector_min(msg, -10, 10)   # ±20 은 좁은 통로에서 옆벽이 잡힘
        self.left = sector_min(msg, 60, 100)
        self.right = sector_min(msg, 260, 300)
        if self.state == 'DRIVE':
            self.drive_step()

    def on_image(self, msg):           # 구독: /camera/image_raw/compressed
        self.last_cam_ns = self.get_clock().now().nanoseconds
        if self.state != 'DRIVE' or self.front >= READ_DIST:
            return                     # 접근 중(READ_DIST 이내)에만 미리 분류
        if self.last_cam_ns - self.last_infer_ns < INFER_PERIOD * 1e9:
            return                     # 디코드 전에 드롭 — scan 콜백 굶기지 않기
        self.last_infer_ns = self.last_cam_ns
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        t0 = time.perf_counter()
        confirmed, self.label, self.same = classify(
            self.model, frame, self.label, self.same, CONF_MIN, NEED)
        self.get_logger().info(f"infer={time.perf_counter() - t0:.2f}s "
                               f"arrow={self.label or 'none'}({self.same}/{NEED})")
        if confirmed:
            self.confirmed = confirmed  # 확정되면 그 이후엔 유지 (래치)

    def drive_step(self):
        front, left, right = self.front, self.left, self.right
        if front >= SLOW_DIST:
            lin = MAX_SPEED
        elif front > WALL_STOP:
            lin = MAX_SPEED * (front - WALL_STOP) / (SLOW_DIST - WALL_STOP)
        else:
            lin = 0.0
        # 벽 유무별 중앙 유지: 양쪽 벽 -> 균형, 한쪽 벽 -> SIDE_TARGET 오프셋 추종, 벽 없음 -> 헤딩만
        # 왼쪽이 가까우면(err<0) 오른쪽으로(ang<0), D항이 지그재그를 감쇠
        lw, rw = left < SIDE_WALL, right < SIDE_WALL
        if lw and rw:
            err = left - right
        elif lw:
            err = left - SIDE_TARGET
        elif rw:
            err = SIDE_TARGET - right
        else:
            err = 0.0
        ang = KP_CENTER * err + KD_CENTER * (err - self.prev_err)
        self.prev_err = err
        if self.head is not None:
            # 절대 헤딩 유지 — 갈림길에서 벽이 사라져도 직진 유지
            ang += KH_HEAD * norm_angle(self.head - self.yaw)
        ang = max(-TURN_SPEED, min(TURN_SPEED, ang))

        # 화살표 확정 + 화살표 벽까지 일정 거리 -> 그 자리에서 90도(backward 는 180도) 회전
        if self.confirmed in TURN_DELTA and front <= TURN_DIST and self.yaw is not None:
            side = {'left': left, 'right': right}.get(self.confirmed, float('inf'))
            if side > SIDE_OPEN:       # 라이다 교차 검증 — 오분류로 벽에 도는 것 방지
                self.enter_turn()
                return
            self.get_logger().warning(
                f"화살표 {self.confirmed} 쪽 막힘(side={side:.2f}) — 래치 해제 후 재분류")
            self.confirmed, self.label, self.same = None, '', 0

        reason = 'DRIVE'
        if front <= WALL_STOP:
            lin, ang, reason = 0.0, 0.0, 'STOP-wall'
        elif self.confirmed not in TURN_DELTA and front <= READ_STOP:
            lin, ang, reason = 0.0, 0.0, 'STOP-wait-arrow'   # 정지하고 계속 분류
            now = self.get_clock().now().nanoseconds
            if self.wait_since_ns is None:
                self.wait_since_ns = now
            elif now - self.wait_since_ns > WAIT_ARROW_TIMEOUT * 1e9:
                # 화살표 장기 미확정 -> 열린 방향으로 fallback (데드락 방지)
                self.confirmed = ('left' if left > SIDE_OPEN else
                                  'right' if right > SIDE_OPEN else 'backward')
                self.get_logger().warning(
                    f"화살표 {WAIT_ARROW_TIMEOUT:.0f}s 미확정 — fallback {self.confirmed}")
        if reason != 'STOP-wait-arrow':
            self.wait_since_ns = None

        self.send(lin, ang)
        self.log(reason, lin, ang)

    def enter_turn(self):
        # 오도메트리 피드백: 목표 yaw 에 도달할 때까지 돌고, 목표 근처에서 감속
        delta = TURN_DELTA[self.confirmed]
        # 현재 yaw 가 아닌 목표 헤딩 기준으로 누적 — 회전마다 오차가 쌓이지 않음
        self.turn_target = norm_angle(self.head + delta)
        # ponytail: /odom 이 끊겨도 영원히 돌지 않게 하는 안전 타임아웃 (여유 3배 + 1초)
        timeout = abs(delta) / TURN_SPEED * 3.0 + 1.0
        self.turn_deadline_ns = self.get_clock().now().nanoseconds + int(timeout * 1e9)
        self.state = 'TURN'
        self.log('TURN-start', 0.0, math.copysign(TURN_SPEED, delta))

    def turn_tick(self):               # 타이머: 0.05초마다 (피드백 회전)
        if self.state != 'TURN':
            return
        err = norm_angle(self.turn_target - self.yaw)
        done = abs(err) < TURN_DONE
        if done or self.get_clock().now().nanoseconds >= self.turn_deadline_ns:
            self.head = self.turn_target    # 새 진행 방향을 기준 헤딩으로
            self.state = 'DRIVE'
            self.confirmed = None
            self.label, self.same = '', 0   # 화살표 확정 상태 리셋
            self.prev_err = 0.0             # PD D항 스파이크 방지
            self.send(0.0, 0.0)
            self.log('DRIVE', 0.0, 0.0)
            return
        # 오차에 비례해 감속(최저 0.4 rad/s) -> 관성 오버슈트 제거
        ang = math.copysign(min(TURN_SPEED, max(0.4, 2.0 * abs(err))), err)
        self.send(0.0, ang)

    def watchdog(self):                 # 타이머: 0.1초마다
        now = self.get_clock().now().nanoseconds
        cam_age = (now - self.last_cam_ns) / 1e9
        scan_age = (now - self.last_scan_ns) / 1e9
        # TURN 은 자체 타임아웃이 있음 — 그동안은 워치독 개입 금지 (STOP 끼어들면 회전이 끊긴다)
        if self.state == 'TURN':
            return
        if scan_age > 0.5:              # scan 없이는 주행 자체가 불가 -> 즉시 정지
            self.get_logger().warning(
                f"/scan {scan_age:.1f}s 미수신 — 정지", throttle_duration_sec=2.0)
            self.send(0.0, 0.0)
        elif cam_age > 0.5:
            # 카메라는 화살표 판독 구간(READ_DIST 이내)에서만 필수 — 멀면 경고만 하고 주행 유지
            self.get_logger().warning(
                f"/camera {cam_age:.1f}s 미수신 — 화살표 판독 불가", throttle_duration_sec=2.0)
            if self.front < READ_DIST:
                self.send(0.0, 0.0)


def main():
    rclpy.init()
    node = MazeRacer()
    try:
        rclpy.spin(node)
    finally:                        # Ctrl+C 로 끝나도 여기는 꼭 실행돼요
        node.send(0.0, 0.0)         # 마지막에 '정지' 를 보내고
        rclpy.spin_once(node, timeout_sec=0.2)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
