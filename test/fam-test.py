import cv2
from collections import Counter
from pupil_apriltags import Detector

# Все интересующие семейства в одном детекторе
FAMILIES = [
    "tag36h11",
    "tag25h9",
    "tagStandard41h12",
    "tagStandard52h13",
    "tagCircle21h7",
    "tagCircle49h12",
]

FAMILIES_STR = ",".join(FAMILIES)


def detect_families_from_frame(frame):
    """
    frame: BGR-кадр (numpy array)
    return: Counter по tag_family
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detector = Detector(families=FAMILIES_STR)
    detections = detector.detect(gray)

    fam_counts = Counter()
    for d in detections:
        fam = d.tag_family.decode("ascii") if isinstance(d.tag_family, bytes) else str(d.tag_family)
        fam_counts[fam] += 1

    return fam_counts, detections


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Камера не открылась")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Не удалось получить кадр с камеры")

    fam_counts, detections = detect_families_from_frame(frame)

    print("Всего детекций:", len(detections))
    if not detections:
        print("Ни одного AprilTag не найдено в этом кадре.")
    else:
        print("По семействам:")
        for fam, cnt in fam_counts.items():
            print(f"  {fam}: {cnt} detections")

    # Показать кадр (чисто чтобы понимать, что анализовали)
    cv2.imshow("Frame used for detection", frame)
    print("Нажми любую клавишу в окне картинки, чтобы закрыть.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
