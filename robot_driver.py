import math
import sys
import time

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

UNSTAMPED = "--unstamped" in sys.argv
MSG = Twist if UNSTAMPED else TwistStamped

SAFE = {'MAX_SPEED': 0.20, 'TURN_SPEED': 1.5}
RACE = {'MAX_SPEED': 0.26, 'TURN_SPEED': 1.5}   # 직선만 최대 — 1.8 rad/s 회전은 관성 밀림으로 위치 틀어짐
PRESET = SAFE if "--safe" in sys.argv else RACE  # 기본 = 최대 속도, --safe 로 감속
MAX_SPEED = PRESET['MAX_SPEED']
TURN_SPEED = PRESET['TURN_SPEED']

WALL_STOP = 0.15   # 화살표 미확정 시 최종 정지선 (TURN_DIST 보다 작아야 회전 지점까지 갈 수 있음)
SLOW_DIST = 1.0       # 여기부터 선형 감속
KP_CENTER = 0.7       # 좌우 벽 거리 차(m) -> angular.z(rad/s) 비례 게인
KD_CENTER = 0.3       # 거리 차 변화율 감쇠 게인 (지그재그 억제)
SIDE_CLAMP = 0.6      # 한쪽이 트인 갈림길에서 중앙 유지 폭주 방지용 클램프
TURN_DELTA = {'left': math.pi / 2, 'right': -math.pi / 2}   # 좌/우만 사용 (backward 미사용)
TURN_DIST = 0.35  # 화살표 확정 상태에서 전방(화살표 벽)이 이 거리면 바로 90도 회전
READ_STOP = 0.45  # 미확정 시 여기서 멈춰 분류 (더 가면 화살표가 화면을 벗어남)
TURN_DONE = 0.05  # 목표 각도 오차(rad)가 이 이내면 회전 종료
ARROW_DIST = 1.5      # 이보다 전방이 멀면 /arrow_dir 무시 (이전 교차로 잔류 메시지 차단)
ARROW_HOLDOFF = 1.5   # 회전 종료 후 이 시간(초) 동안 /arrow_dir 무시 (재래치 방지)
STEER_FADE = 0.7      # 전방이 이보다 가까우면 중앙 유지 조향을 선형으로 줄여 직진 접근
APPROACH_MIN = 0.10   # 화살표 확정 후 회전 지점까지 기어가지 않게 하는 최저 접근 속도
SIDE_GUARD = 0.25     # 회전 직후 홀드오프 중에도 이보다 벽에 붙으면 밀어냄 (벽 긁기 방지)
SIDE_OPEN = 0.45      # 회전 전 그쪽이 실제로 뚫려 있는지 LiDAR 검증 (오인식 벽 돌진 방지)
KP_YAW = 1.2          # 기준 방위(복도 축) 이탈 보정 게인 — 좌편향 등 하드웨어 편차 보정 노브


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


class RobotDriver(Node):
    def __init__(self):
        super().__init__("robot_driver")
        self.pub = self.create_publisher(MSG, "/cmd_vel", 10)
        self.create_subscription(String, "/arrow_dir", self.on_arrow, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, qos_profile_sensor_data)
        self.create_timer(0.1, self.watchdog)
        self.create_timer(0.05, self.turn_tick)   # 회전 제어 전용 (20Hz)
        now = self.get_clock().now().nanoseconds
        self.last_scan_ns = now
        self.state = 'DRIVE'
        self.front = self.left = self.right = float('inf')
        self.yaw = None                # /odom 수신 전엔 회전 불가
        self.turn_target = 0.0
        self.turn_deadline_ns = 0      # 피드백 회전 안전 타임아웃
        self.confirmed = None
        self.prev_err = 0.0
        self.arrow_ignore_until_ns = 0   # 회전 직후 잔류 화살표 무시 기한
        self.grid_ref = None             # 격자 기준 방위 (첫 odom yaw = 시작 복도 방향)

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
            f"arrow={self.confirmed or 'none'} "
            f"lin={lin:.2f} ang={ang:.2f}")

    def on_odom(self, msg):            # 구독: /odom (쿼터니언 -> yaw)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.grid_ref is None:        # 시작 자세가 복도와 평행하다고 가정
            self.grid_ref = self.yaw

    def grid_heading(self):
        # 현재 yaw 에서 가장 가까운 90도 격자 방향 — 손으로 옮겨 놓아도 자동 복구, 오차 최대 45도
        k = round(norm_angle(self.yaw - self.grid_ref) / (math.pi / 2))
        return norm_angle(self.grid_ref + k * math.pi / 2)

    def on_scan(self, msg):            # 구독: /scan
        self.last_scan_ns = self.get_clock().now().nanoseconds
        self.front = sector_min(msg, -10, 10)   # ±20 은 좁은 통로에서 옆벽이 잡힘
        # 90/270 중심 대칭 섹터 — 60/300 은 앞으로 치우쳐 전방 벽 모서리가 좌우 오차를 오염시킴
        self.left = sector_min(msg, 70, 110)
        self.right = sector_min(msg, 250, 290)
        if self.state == 'DRIVE':
            self.drive_step()

    def on_arrow(self, msg):
        if msg.data not in TURN_DELTA or self.state != 'DRIVE':
            return
        if self.get_clock().now().nanoseconds < self.arrow_ignore_until_ns:
            return                      # 회전 직후 큐에 남은 이전 교차로 화살표 무시
        if self.front > ARROW_DIST:
            return                      # 벽 근처가 아니면 잔류/오인식 메시지
        if self.confirmed != msg.data:
            self.get_logger().info(f"arrow latched: {msg.data} (front={self.front:.2f})")
        self.confirmed = msg.data       # 래치

    def drive_step(self):
        front, left, right = self.front, self.left, self.right
        if front >= SLOW_DIST:
            lin = MAX_SPEED
        elif front > WALL_STOP:
            lin = MAX_SPEED * (front - WALL_STOP) / (SLOW_DIST - WALL_STOP)
        else:
            lin = 0.0
        # 회전 직후 홀드오프: 교차로 코너의 좌우 비대칭(한쪽 벽/한쪽 개방)을 중앙 유지가
        # "치우침"으로 오독해 로봇을 되돌리므로, 이 구간은 격자 방위 고정만으로 직진
        if self.get_clock().now().nanoseconds < self.arrow_ignore_until_ns:
            self.prev_err = 0.0
            # 벽에 붙은 채 회전이 끝난 경우만 밀어내기 (중앙 유지 오독은 계속 차단)
            if left < SIDE_GUARD:
                ang = -KP_CENTER * (SIDE_GUARD - left)
            elif right < SIDE_GUARD:
                ang = KP_CENTER * (SIDE_GUARD - right)
            else:
                ang = 0.0
        else:
            # PD 벽 중앙 유지: 왼쪽이 가까우면(err<0) 오른쪽으로(ang<0), D항이 지그재그를 감쇠
            err = min(left, SIDE_CLAMP) - min(right, SIDE_CLAMP)
            ang = KP_CENTER * err + KD_CENTER * (err - self.prev_err)
            self.prev_err = err
            if front < STEER_FADE:      # 벽/교차로 접근 중엔 조향을 줄여 직진 유지 (벽에 붙는 문제)
                ang *= max(0.0, (front - TURN_DIST) / (STEER_FADE - TURN_DIST))
        # 방위 고정: 복도 축에서 틀어진 각도를 되돌림 (좌편향 하드웨어 편차 보정, 페이드 미적용)
        if self.yaw is not None and self.grid_ref is not None:
            ang += KP_YAW * norm_angle(self.grid_heading() - self.yaw)
        ang = max(-TURN_SPEED, min(TURN_SPEED, ang))   # 조향 총량 클램프

        # 화살표 확정 + 화살표 벽까지 일정 거리 -> 그 자리에서 90도 회전
        if self.confirmed in TURN_DELTA and front <= TURN_DIST and self.yaw is not None:
            side = left if self.confirmed == 'left' else right
            if side > SIDE_OPEN:        # 그쪽이 실제로 뚫려 있을 때만 회전 (주최 코드에서 이식)
                self.enter_turn()
                return
            self.get_logger().warning(
                f"arrow={self.confirmed} 인데 그쪽 벽 {side:.2f}m — 오인식 의심, 재분류 대기")
            self.confirmed = None       # 래치 해제 -> STOP-wait-arrow 로 재확정 대기
        if self.confirmed in TURN_DELTA:
            lin = max(lin, APPROACH_MIN)   # 확정됐으면 회전 지점까지 기어가지 말고 바로 접근

        reason = 'DRIVE'
        if front <= WALL_STOP:
            lin, ang, reason = 0.0, 0.0, 'STOP-wall'
        elif self.confirmed not in TURN_DELTA and front <= READ_STOP:
            lin, ang, reason = 0.0, 0.0, 'STOP-wait-arrow'   # 정지하고 계속 분류

        self.send(lin, ang)
        self.log(reason, lin, ang)

    def enter_turn(self):
        # 오도메트리 피드백: 목표 yaw 에 도달할 때까지 돌고, 목표 근처에서 감속
        delta = TURN_DELTA[self.confirmed]
        # 현재 yaw 가 아닌 격자 기준 방위에서 90도 -> 누적된 틀어짐이 회전 때마다 리셋됨
        self.turn_target = norm_angle(self.grid_heading() + delta)
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
            if not done:   # 벽에 걸려 회전이 목표각에 못 미친 채 끝남 — 필드 진단용
                self.get_logger().warning(f"TURN timeout: 잔여 오차 {math.degrees(err):.0f}도")
            self.state = 'DRIVE'
            self.confirmed = None
            self.prev_err = 0.0             # PD D항 스파이크 방지
            self.arrow_ignore_until_ns = (self.get_clock().now().nanoseconds
                                          + int(ARROW_HOLDOFF * 1e9))
            self.send(0.0, 0.0)
            self.log('DRIVE', 0.0, 0.0)
            return
        # 오차에 비례해 감속(최저 0.4 rad/s) -> 관성 오버슈트 제거
        ang = math.copysign(min(TURN_SPEED, max(0.4, 2.0 * abs(err))), err)
        self.send(0.0, ang)

    def watchdog(self):                 # 타이머: 0.1초마다
        now = self.get_clock().now().nanoseconds
        scan_age = (now - self.last_scan_ns) / 1e9
        # TURN 은 자체 타임아웃이 있음 — 그동안은 워치독 개입 금지 (STOP 끼어들면 회전이 끊긴다)
        if self.state == 'TURN':
            return
        if scan_age > 0.5:              # scan 없이는 주행 자체가 불가 -> 즉시 정지
            self.get_logger().warning(
                f"/scan {scan_age:.1f}s 미수신 — 정지", throttle_duration_sec=2.0)
            self.send(0.0, 0.0)


def main():
    # 기본 SIGINT 핸들러는 Ctrl+C 즉시 컨텍스트를 죽여 finally 의 정지 발행이 무효가 됨
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = RobotDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(3):          # 정지를 여러 번 발행해 확실히 전달
            node.send(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
