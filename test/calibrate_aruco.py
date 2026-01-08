#!/usr/bin/env python3
"""
Калибровка камеры по ArUco GridBoard.
Результат: camera_matrix.npy и dist_coeffs.npy в текущей папке.
"""

from pathlib import Path

import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_1000" 

# размер сетки маркеров на распечатке:
BOARD_COLS = 6   # маркеров по горизонтали  (X)
BOARD_ROWS = 6   # маркеров по вертикали    (Y)

# физический размер (в метрах)
MARKER_LENGTH = 0.04       # длина стороны маркера, например 4 см -> 0.04
MARKER_SEPARATION = 0.01   # расстояние между маркерами, например 1 см -> 0.01

CAMERA_MATRIX_PATH = Path("camera_matrix.npy")
DIST_COEFFS_PATH = Path("dist_coeffs.npy")


def main():
    aruco = cv2.aruco

    if not hasattr(aruco, ARUCO_DICT_NAME):
        raise ValueError(f"Словарь {ARUCO_DICT_NAME} не поддерживается")

    dict_id = getattr(aruco, ARUCO_DICT_NAME)
    dictionary = aruco.getPredefinedDictionary(dict_id)

    board = aruco.GridBoard(
        (BOARD_COLS, BOARD_ROWS),
        MARKER_LENGTH,
        MARKER_SEPARATION,
        dictionary,
    )

    all_corners = []
    all_ids = []
    image_size = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Камера не открылась")

    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, detector_params)

    print("Калибровка камеры.")
    print("Требование: в кадре должно быть минимум 5 маркеров.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame, corners, ids)

        cv2.putText(
            frame,
            "SPACE = capture, ESC = finish",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32 and ids is not None and len(ids) > 0:  # SPACE
            num_markers = 0 if ids is None else len(ids)
            if ids is None or num_markers < 1:
                print(f"Слишком мало маркеров в кадре ({num_markers} < 1) — кадр не сохраняю")
                continue
            print(f"Сохраняем кадр, найдено маркеров: {len(ids)}")
            all_corners.append(corners)
            all_ids.append(ids)
            image_size = gray.shape[::-1]

    cap.release()
    cv2.destroyAllWindows()

    if len(all_corners) < 10:
        print("Мало кадров для калибровки (нужно хотя бы 10–20).")
        return

    print("Считаем калибровку, кадров:", len(all_corners))

    all_corners_concat = []
    all_ids_concat = []
    marker_counter_per_frame = []

    for corners, ids in zip(all_corners, all_ids):
        marker_counter_per_frame.append(len(ids))
        all_corners_concat.extend(corners)
        all_ids_concat.extend(ids.reshape(-1))

    all_ids_concat = np.array(all_ids_concat, dtype=np.int32)
    marker_counter_per_frame = np.array(marker_counter_per_frame, dtype=np.int32)

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraAruco(
        all_corners_concat,
        all_ids_concat,
        marker_counter_per_frame,
        board,
        image_size,
        None,
        None,
    )

    print("Средняя ошибка переотображения (reprojection error):", ret)
    print("camera_matrix:\n", camera_matrix)
    print("dist_coeffs:\n", dist_coeffs.ravel())

    np.save(CAMERA_MATRIX_PATH, camera_matrix)
    np.save(DIST_COEFFS_PATH, dist_coeffs)

    print(f"Сохранено в {CAMERA_MATRIX_PATH} и {DIST_COEFFS_PATH}")


if __name__ == "__main__":
    main()
