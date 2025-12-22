#!/usr/bin/env python3
"""
Простой демо-скрипт:
- Захватывает видео с камеры
- Детектит ArUco-маркеры
- Немного сглаживает положение маркеров
- (опционально) рисует 3D-оси, если есть калибровка камеры
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ====== НАСТРОЙКИ ПОД СЕБЯ ======

# Имя словаря ArUco (подставь то, который у тебя реально сработал)
ARUCO_DICT_NAME = "DICT_4X4_1000"

# Номер камеры (0 — встроенная / первая USB)
CAMERA_ID = 0

# Физический размер маркера в метрах (если нужен pose)
MARKER_LENGTH_METERS = 0.04  # 4 см

# Пути к файлам калибровки (если пока нет — файлы можно не создавать)
CAMERA_MATRIX_PATH = Path("camera_matrix.npy")
DIST_COEFFS_PATH = Path("dist_coeffs.npy")

# Коэффициент сглаживания (0.1 — плавно, 0.5 — быстро реагирует)
SMOOTHING_ALPHA = 0.3


# ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======

def load_camera_params() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Пытаемся загрузить параметры камеры из .npy.
    Если файлов нет — возвращаем (None, None).
    """
    if CAMERA_MATRIX_PATH.exists() and DIST_COEFFS_PATH.exists():
        camera_matrix = np.load(CAMERA_MATRIX_PATH)
        dist_coeffs = np.load(DIST_COEFFS_PATH)
        return camera_matrix, dist_coeffs
    return None, None


def create_aruco_detector(dict_name: str) -> Tuple[cv2.aruco.ArucoDetector, int]:
    """Создаём детектор ArUco с адекватными параметрами."""
    aruco = cv2.aruco

    if not hasattr(aruco, dict_name):
        raise ValueError(f"Словарь {dict_name} не поддерживается этой версией OpenCV")

    dict_id = getattr(aruco, dict_name)
    dictionary = aruco.getPredefinedDictionary(dict_id)

    params = aruco.DetectorParameters()

    # Улучшаем точность углов
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01

    # Можно поджать фильтрацию по размеру, если много мусора
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0

    detector = aruco.ArucoDetector(dictionary, params)
    return detector, dict_id


def ema(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    """Экспоненциальное сглаживание (EMA) для вектора."""
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


# ====== ОСНОВНОЙ ЦИКЛ ======

def main() -> None:
    # 1. Загружаем калибровку камеры (если есть)
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None or dist_coeffs is None:
        print("⚠️  Калибровка не найдена: pose будет пропущен, только 2D-рамки.")
    else:
        print("✅ Загружены параметры камеры, pose/оси будут рисоваться.")

    # 2. Создаём детектор
    detector, dict_id = create_aruco_detector(ARUCO_DICT_NAME)
    print(f"Используем словарь ArUco: {ARUCO_DICT_NAME} (id={dict_id})")

    # 3. Открываем камеру
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру с id={CAMERA_ID}")

    # Словарь для сглаживания центров и поз
    smoothed_centers: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)
    smoothed_tvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    aruco = cv2.aruco

    print("Нажми 'q' в окне видео, чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Кадр не получен, выходим.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4. Детектим маркеры
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            ids = ids.reshape(-1)

            # Рисуем рамки
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 4.1. Сглаживаем центры и (опционально) pose
            for i, marker_id in enumerate(ids):
                pts = corners[i][0]  # shape (4, 2)
                center = pts.mean(axis=0)  # (x, y)

                # Сглаживаем 2D-центр
                smoothed = ema(smoothed_centers[marker_id], center, SMOOTHING_ALPHA)
                smoothed_centers[marker_id] = smoothed

                cx, cy = int(smoothed[0]), int(smoothed[1])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    f"id={marker_id}",
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # 4.2. Оценка позы, если есть калибровка
            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _obj_points = aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
                )

                for marker_id, rvec, tvec in zip(ids, rvecs, tvecs):
                    tvec = tvec.reshape(-1)
                    smoothed_t = ema(smoothed_tvecs[marker_id], tvec, SMOOTHING_ALPHA)
                    smoothed_tvecs[marker_id] = smoothed_t

                    # Рисуем 3D-ось по сглаженному положению
                    aruco.drawAxis(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        smoothed_t,
                        MARKER_LENGTH_METERS * 0.5,
                    )

                    # Выведем в консоль расстояние до маркера
                    dist_m = np.linalg.norm(smoothed_t)
                    print(f"id={marker_id}: distance ≈ {dist_m:.3f} m", end="\r")

        cv2.imshow("ArUco demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
