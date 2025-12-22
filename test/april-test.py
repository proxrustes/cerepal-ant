import cv2
from pupil_apriltags import Detector

FAMILIES = [
    "tag36h11",
    "tag25h9",
    "tagStandard41h12",
    "tagStandard52h13",
    "tagCircle21h7",
    "tagCircle49h12",
]

# Инициализируем детектор
detector = Detector(
    families="tagCircle49h12",
    nthreads=2,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)

cap = cv2.VideoCapture(0)  # 0 — первая камера

if not cap.isOpened():
    raise RuntimeError("Камера не открылась")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    print("len(detections) =", len(detections))
    for d in detections:
        print("id:", d.tag_id, "center:", d.center)


    # Рисуем рамки и id поверх картинки
    for d in detections:
        corners = d.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cx, cy = map(int, d.center)
        cv2.putText(frame, str(d.tag_id), (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AprilTag", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
