"""로봇 카메라로 화살표 데이터 수집 + 텔레옵 (PC에서 실행).

주의: robot_driver.py 를 끈 상태에서 실행할 것 (/cmd_vel 충돌).

  주행:  W 전진 / X 후진 / A 좌회전 / D 우회전 / S·스페이스 정지
  저장:  L = left, R = right   (각 100장, 80/20 자동 분배, rc_ 접두사)
  종료:  Q
"""
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

DATA = Path("arrow_dataset")
LABELS = ("left", "right")
TARGET = 100
LIN, ANG = 0.15, 0.8          # 텔레옵 속도 (수집용이라 낮게)
DRIVE = {ord("w"): (LIN, 0.0), ord("x"): (-LIN, 0.0),
         ord("a"): (0.0, ANG), ord("d"): (0.0, -ANG),
         ord("s"): (0.0, 0.0), ord(" "): (0.0, 0.0)}
SAVE = {ord("l"): "left", ord("r"): "right"}


def count(label):             # 로봇캠(rc_) 사진만 센다 — 기존 웹캠 사진과 구분
    return sum(len(list((DATA / s / label).glob("rc_*.jpg")))
               for s in ("train", "val"))


class Collector(Node):
    def __init__(self):
        super().__init__("collect_arrow_data")
        self.frame = None
        self.pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.create_subscription(CompressedImage, "/camera/image_raw/compressed",
                                 self.on_image, qos_profile_sensor_data)
        self.cmd = (0.0, 0.0)

    def on_image(self, msg):
        f = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if f is not None:
            self.frame = f

    def send(self):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.twist.linear.x, m.twist.angular.z = self.cmd
        self.pub.publish(m)


def main():
    rclpy.init()
    node = Collector()
    print(__doc__)
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.03)
            if node.frame is None:
                continue
            frame = node.frame
            h, w = frame.shape[:2]
            s = int(min(h, w) * 0.7)      # 분류기와 동일한 중앙 70% 크롭
            x, y = (w - s) // 2, (h - s) // 2
            crop = frame[y:y + s, x:x + s]
            view = frame.copy()
            cv2.rectangle(view, (x, y), (x + s, y + s), (0, 255, 0), 2)
            for i, name in enumerate(LABELS):
                cv2.putText(view, f"{name}: {count(name)}/{TARGET}", (16, 30 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(view, f"cmd: {node.cmd[0]:+.2f} {node.cmd[1]:+.2f}",
                        (16, 30 + len(LABELS) * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.imshow("collect (WXAD 주행 / L R 저장 / Q 종료)", view)

            key = cv2.waitKey(1) & 0xFF
            if 65 <= key <= 90:            # CapsLock/Shift 대문자 -> 소문자
                key += 32
            if key == ord("q"):
                break
            if key in DRIVE:
                node.cmd = DRIVE[key]
            label = SAVE.get(key)
            if label and count(label) < TARGET:
                n = count(label)
                split = "train" if n < TARGET * 0.8 else "val"   # 80 / 20 자동 분배
                path = DATA / split / label / f"rc_{label}_{n:03d}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), crop)
                print(f"saved {path} ({n + 1}/{TARGET})")
            node.send()                    # 현재 주행 명령을 매 루프 발행
    finally:
        node.cmd = (0.0, 0.0)
        for _ in range(3):                 # 종료 시 정지 확실히 전달
            node.send()
            time.sleep(0.05)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
