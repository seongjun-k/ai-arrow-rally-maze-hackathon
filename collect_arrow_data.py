from pathlib import Path
import cv2

DATA = Path("arrow_dataset")
KEYS = {ord("f"): "forward", ord("l"): "left",
        ord("r"): "right", ord("b"): "backward",
        ord("o"): "obstacle"}              # ← O 키 = 장애물
TARGET = 100

def count(label):                 # 지금까지 모은 장수 (train + val)
    return sum(len(list((DATA / s / label).glob("*.jpg")))
               for s in ("train", "val"))

cam = cv2.VideoCapture(0)
while True:
    ok, frame = cam.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    s = int(min(h, w) * 0.7)          # 화면 가운데 70% = 초록 상자
    x, y = (w - s) // 2, (h - s) // 2
    crop = frame[y:y + s, x:x + s]
    view = frame.copy()
    cv2.rectangle(view, (x, y), (x + s, y + s), (0, 255, 0), 2)
    for i, name in enumerate(KEYS.values()):
        cv2.putText(view, f"{name}: {count(name)}/{TARGET}", (16, 30 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.imshow("collect", view)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    label = KEYS.get(key)
    if label and count(label) < TARGET:
        n = count(label)
        split = "train" if n < TARGET * 0.8 else "val"   # 80 / 20 자동 분배
        path = DATA / split / label / f"{label}_{n:02d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)
        print(f"saved {path} ({n + 1}/{TARGET})")
cam.release()
cv2.destroyAllWindows()
