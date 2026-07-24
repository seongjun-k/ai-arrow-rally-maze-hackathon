import shutil
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")            # 분류(cls)용 작은 모델에서 출발
results = model.train(
    data="arrow_dataset",                 # train/ val/ 폴더가 있는 곳 (left/right 2클래스)
    epochs=80, imgsz=224, patience=20,
    fliplr=0.0, flipud=0.0,               # 반전 증강 금지 — 좌/우가 뒤집혀요!
    auto_augment=None, erasing=0.0,
    degrees=10.0, translate=0.1, scale=0.5,   # 접근 거리(0.3~1.5m)에 따른 크기/위치 변화 대응
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.5)        # 조명 변화 대응
best = Path(results.save_dir) / "weights" / "best.pt"
shutil.copy2(best, "arrow_classifier.pt")
print("Model ready: arrow_classifier.pt")
