import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, LaserScan
from std_msgs.msg import String
from ultralytics import YOLO

READ_DIST = 1.0       # 여기부터 화살표 분류 시작 — 1.5m 는 화살표가 작아 좌/우 오인식 발생(실측)
INFER_PERIOD = 0.05   # YOLO 추론 최소 간격(초) — 추론 실측 ~5ms, 카메라 fps가 실질 상한 (빠른 확정)
CONF_MIN = 0.90       # 확신도 기준 — 벽 무늬 오인식 방지 상향
NEED = 3              # 같은 방향 연속 프레임 수
USE_LABELS = ('left', 'right')   # 이 라벨만 사용 — backward/front 클래스는 무시


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
    conf = float(r.probs.top1conf)
    if conf < conf_min or name not in USE_LABELS:   # 확신도 낮음/미사용 라벨 프레임은
        return None, last, same, conf, name          # 무시하고 진행 상황 유지
    same = same + 1 if name == last else 1          # 같은 방향 연속 세기
    confirmed = name if same >= need else None      # need 번 연속이어야 확정
    return confirmed, name, same, conf, name


class ArrowClassifierNode(Node):
    def __init__(self):
        super().__init__("arrow_classifier_node")
        self.model = YOLO("arrow_classifier.pt")
        # 워밍업: 첫 추론이 ~2.2초 걸려 executor 를 막음(실측) — 주행 전에 미리 1회 실행
        self.model.predict(np.zeros((224, 224, 3), np.uint8), verbose=False)
        self.pub = self.create_publisher(String, "/arrow_dir", 10)
        self.create_subscription(CompressedImage, "/camera/image_raw/compressed",
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "/scan", self.on_scan, qos_profile_sensor_data)
        self.front = float('inf')
        self.label, self.same = '', 0
        self.last_cam_ns = 0
        self.last_infer_ns = 0

    def on_scan(self, msg):            # 구독: /scan
        self.front = sector_min(msg, -10, 10)   # ±20 은 좁은 통로에서 옆벽이 잡힘

    def on_image(self, msg):           # 구독: /camera/image_raw/compressed
        self.last_cam_ns = self.get_clock().now().nanoseconds
        if self.front >= READ_DIST:
            self.label, self.same = '', 0   # 교차로 벗어나면 확정 카운트 초기화
            return
        if self.last_cam_ns - self.last_infer_ns < INFER_PERIOD * 1e9:
            return                     # 디코드 전에 드롭 — scan 콜백 굶기지 않기
        self.last_infer_ns = self.last_cam_ns
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return
        t0 = time.perf_counter()
        confirmed, self.label, self.same, conf, raw = classify(
            self.model, frame, self.label, self.same, CONF_MIN, NEED)
        self.get_logger().info(f"infer={time.perf_counter() - t0:.2f}s "
                               f"arrow={self.label or 'none'}({self.same}/{NEED}) "
                               f"raw={raw} conf={conf:.2f}")
        if confirmed:
            out = String()
            out.data = confirmed
            self.pub.publish(out)
            self.get_logger().info(f"arrow_dir 발행: {confirmed}")


def main():
    rclpy.init()
    node = ArrowClassifierNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
