import cv2

# Все словари, которые хотим проверить
ARUCO_DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]


def detect_aruco_for_all_dicts(gray):
    """
    gray: grayscale image (numpy array)
    return: dict { dict_name: (corners, ids) }
    """
    results = {}

    aruco = cv2.aruco
    parameters = aruco.DetectorParameters()

    for name in ARUCO_DICT_NAMES:
        # пропускаем, если в данной версии OpenCV словаря нет
        if not hasattr(aruco, name):
            continue

        dict_id = getattr(aruco, name)
        dictionary = aruco.getPredefinedDictionary(dict_id)
        detector = aruco.ArucoDetector(dictionary, parameters)

        corners, ids, rejected = detector.detectMarkers(gray)
        results[name] = (corners, ids)

    return results


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Камера не открылась")

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Не удалось получить кадр")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    results = detect_aruco_for_all_dicts(gray)

    print("Результаты по словарям:")
    for name, (corners, ids) in results.items():
        n = 0 if ids is None else len(ids)
        print(f"{name}: {n} detections")

    cv2.imshow("Frame used for detection", frame)
    print("Нажми любую клавишу в окне с картинкой, чтобы закрыть.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
