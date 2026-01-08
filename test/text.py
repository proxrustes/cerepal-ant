import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_1000"  # <- твой словарь
MARKER_ID = 524                    # id маркера, за которым следим

aruco = cv2.aruco
dict_id = getattr(aruco, ARUCO_DICT_NAME)
dictionary = aruco.getPredefinedDictionary(dict_id)

params = aruco.DetectorParameters()
params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX

detector = aruco.ArucoDetector(dictionary, params)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Камера не открылась")

centers = []

print("Держи доску как можно неподвижнее. Ждём ~3 сек...")

for i in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)
    if ids is None:
        continue

    ids = ids.reshape(-1)
    for idx, mid in enumerate(ids):
        if mid == MARKER_ID:
            pts = corners[idx][0]
            center = pts.mean(axis=0)
            centers.append(center)
            break

cap.release()

centers = np.array(centers)
if len(centers) == 0:
    print("Не нашёл указанный маркер, попробуй другой MARKER_ID.")
else:
    diffs = centers - centers.mean(axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    print(f"Кадров с маркером: {len(centers)}")
    print(f"Средняя дрожь (пиксели): {dist.mean():.3f}")
    print(f"Максимальная дрожь (пиксели): {dist.max():.3f}")
