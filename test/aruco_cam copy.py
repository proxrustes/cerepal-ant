#!/usr/bin/env python3
"""
ArUco demo + Kalman filter (snappy) + hold при пропаже маркера.

- Детект ArUco
- 2D центр маркера фильтруется Kalman (x,y,vx,vy)
- 3D tvec фильтруется Kalman (x,y,z,vx,vy,vz) если есть калибровка
- rvec не фильтруем: используем последний видимый (ок для коротких пропаж)
- Если маркер пропал на HOLD_SECONDS — рисуем предсказание (predict) и оси по last_rvec
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import time 
import json
# ====== НАСТРОЙКИ ПОД СЕБЯ ======

ARUCO_DICT_NAME = "DICT_4X4_1000"
CAMERA_ID = 0

# Физический размер маркера в метрах (для pose)
MARKER_LENGTH_METERS = 0.03797

CAMERA_MATRIX_PATH = Path("camera_matrix.npy")
DIST_COEFFS_PATH = Path("dist_coeffs.npy")

# EMA сглаживание
SMOOTHING_ALPHA = 0.3

# "Не терять маркер" если он пропал на короткое время
HOLD_SECONDS = 2.0       # держать последнюю позу/центр
FORGET_SECONDS = 5.0     # потом полностью забыть маркер


# ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======

def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len: float) -> None:
    rvec = np.asarray(rvec).reshape(3, 1)
    tvec = np.asarray(tvec).reshape(3, 1)

    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len)
    else:
        # в этой сборке не умеем рисовать оси
        pass


def load_camera_params() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Пытаемся загрузить параметры камеры из .npy.
    Если файлов нет — возвращаем (None, None).
    """
    if CAMERA_MATRIX_PATH.exists() and DIST_COEFFS_PATH.exists():
        camera_matrix = np.load(CAMERA_MATRIX_PATH)
        dist_coeffs = np.load(DIST_COEFFS_PATH)
        return camera_matrix, dist_coeffs
    return None, None


def create_aruco_detector(dict_name: str):
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
    # 1) Загружаем калибровку камеры (если есть)
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None or dist_coeffs is None:
        print("⚠️  Калибровка не найдена: pose будет пропущен, только 2D-рамки.")
    else:
        print("✅ Загружены параметры камеры, pose/оси будут рисоваться.")

    # 2) Создаём детектор
    detector, dict_id = create_aruco_detector(ARUCO_DICT_NAME)
    print(f"Используем словарь ArUco: {ARUCO_DICT_NAME} (id={dict_id})")

    # 3) Открываем камеру
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру с id={CAMERA_ID}")

    # Пытаемся узнать FPS (часто возвращает 0/NaN) — тогда считаем 30
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30.0

    HOLD_FRAMES = int(round(HOLD_SECONDS * fps))
    FORGET_FRAMES = int(round(FORGET_SECONDS * fps))

    # Словари для сглаживания и "памяти"
    smoothed_centers: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)
    smoothed_tvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    last_seen_frame: Dict[int, int] = {}
    last_rvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)
    last_tvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    aruco = cv2.aruco
    frame_idx = 0

    print("Нажми 'q' в окне видео, чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Кадр не получен, выходим.")
            break

        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 4) Детектим маркеры
        corners, ids, rejected = detector.detectMarkers(gray)

        seen_ids = set()

        if ids is not None and len(ids) > 0:
            ids = ids.reshape(-1)
            seen_ids = set(int(x) for x in ids)

            # Рисуем рамки
            aruco.drawDetectedMarkers(frame, corners, ids)

            # 4.1) Сглаживаем центры
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id)
                last_seen_frame[marker_id] = frame_idx

                pts = corners[i][0]  # shape (4, 2)
                center = pts.mean(axis=0)  # (x, y)

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

            # 4.2) Оценка позы, если есть калибровка
            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _obj_points = aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
                )

                for marker_id, rvec, tvec in zip(ids, rvecs, tvecs):
                    marker_id = int(marker_id)

                    tvec = tvec.reshape(-1)
                    smoothed_t = ema(smoothed_tvecs[marker_id], tvec, SMOOTHING_ALPHA)
                    smoothed_tvecs[marker_id] = smoothed_t

                    # сохраняем последнюю позу
                    last_rvecs[marker_id] = rvec
                    last_tvecs[marker_id] = smoothed_t
                 
                    # Рисуем оси
                    draw_axes(
                        frame,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        smoothed_t,
                        MARKER_LENGTH_METERS * 0.5,
                    )

                    dist_m = float(np.linalg.norm(smoothed_t))
                    print(f"id={marker_id}: distance ≈ {dist_m:.3f} m", end="\r")

        # 5) Дорисовываем "пропавшие, но ещё живые" (hold)
        to_delete = []
        for marker_id, last_f in list(last_seen_frame.items()):
            age = frame_idx - last_f

            if age > FORGET_FRAMES:
                to_delete.append(marker_id)
                continue

            # если в этом кадре он виден — уже нарисовали
            if marker_id in seen_ids:
                continue

            # если пропал недавно — рисуем последнюю позицию
            if age <= HOLD_FRAMES:
                c = smoothed_centers.get(marker_id)
                if c is not None:
                    cx, cy = int(c[0]), int(c[1])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)  # жёлтый = held
                    cv2.putText(
                        frame,
                        f"id={marker_id} (held)",
                        (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

                if camera_matrix is not None and dist_coeffs is not None:
                    rvec = last_rvecs.get(marker_id)
                    tvec = last_tvecs.get(marker_id)
                    if rvec is not None and tvec is not None:
                        draw_axes(
                            frame,
                            camera_matrix,
                            dist_coeffs,
                            rvec,
                            tvec,
                            MARKER_LENGTH_METERS * 0.5,
                        )

        # чистим старые маркеры
        for marker_id in to_delete:
            last_seen_frame.pop(marker_id, None)
            smoothed_centers.pop(marker_id, None)
            smoothed_tvecs.pop(marker_id, None)
            last_rvecs.pop(marker_id, None)
            last_tvecs.pop(marker_id, None)

        cv2.imshow("ArUco demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()




def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
    t = np.asarray(tvec).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T  # T_C<-T

