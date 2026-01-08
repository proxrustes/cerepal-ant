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


# ====== НАСТРОЙКИ ======

ARUCO_DICT_NAME = "DICT_4X4_1000"
CAMERA_ID = 0
MARKER_LENGTH_METERS = 0.085

CAMERA_MATRIX_PATH = Path("camera_matrix.npy")
DIST_COEFFS_PATH = Path("dist_coeffs.npy")

# Hold/forget
HOLD_SECONDS = 1.0
FORGET_SECONDS = 5.0

# Kalman включатели
USE_KALMAN_CENTER = True
USE_KALMAN_TVEC = True  # работает только если есть калибровка

# Kalman тюнинг (подбирается)
# Чем больше PROCESS_NOISE_* -> быстрее реагирует (менее "липкий"), но больше дрожит
# Чем больше MEAS_NOISE_* -> меньше верит измерениям (больше сглаживает), но больше лага
CENTER_PROCESS_NOISE_POS = 1e-2
CENTER_PROCESS_NOISE_VEL = 1e-1
CENTER_MEAS_NOISE = 2.0  # пиксели (дисперсия, условно)

TVEC_PROCESS_NOISE_POS = 1e-4
TVEC_PROCESS_NOISE_VEL = 1e-2
TVEC_MEAS_NOISE = 1e-3  # метры (дисперсия, условно)


# ====== ВСПОМОГАТЕЛЬНЫЕ ======

def draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len: float) -> None:
    rvec = np.asarray(rvec).reshape(3, 1)
    tvec = np.asarray(tvec).reshape(3, 1)
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len)


def load_camera_params() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if CAMERA_MATRIX_PATH.exists() and DIST_COEFFS_PATH.exists():
        return np.load(CAMERA_MATRIX_PATH), np.load(DIST_COEFFS_PATH)
    return None, None


def create_aruco_detector(dict_name: str):
    aruco = cv2.aruco
    if not hasattr(aruco, dict_name):
        raise ValueError(f"Словарь {dict_name} не поддерживается этой версией OpenCV")

    dict_id = getattr(aruco, dict_name)
    dictionary = aruco.getPredefinedDictionary(dict_id)

    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.01
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.0

    detector = aruco.ArucoDetector(dictionary, params)
    return detector, dict_id


def make_kalman_2d(dt: float, q_pos: float, q_vel: float, r_meas: float) -> cv2.KalmanFilter:
    """
    State: [x, y, vx, vy]
    Meas:  [x, y]
    """
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array(
        [[1, 0, dt, 0],
         [0, 1, 0, dt],
         [0, 0, 1,  0],
         [0, 0, 0,  1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], dtype=np.float32
    )

    kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([r_meas, r_meas]).astype(np.float32)
    kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0
    kf.statePost = np.zeros((4, 1), dtype=np.float32)
    return kf


def make_kalman_3d(dt: float, q_pos: float, q_vel: float, r_meas: float) -> cv2.KalmanFilter:
    """
    State: [x, y, z, vx, vy, vz]
    Meas:  [x, y, z]
    """
    kf = cv2.KalmanFilter(6, 3)

    kf.transitionMatrix = np.array(
        [[1, 0, 0, dt, 0,  0],
         [0, 1, 0, 0,  dt, 0],
         [0, 0, 1, 0,  0,  dt],
         [0, 0, 0, 1,  0,  0],
         [0, 0, 0, 0,  1,  0],
         [0, 0, 0, 0,  0,  1]], dtype=np.float32
    )
    kf.measurementMatrix = np.array(
        [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0]], dtype=np.float32
    )

    kf.processNoiseCov = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)
    kf.measurementNoiseCov = np.diag([r_meas, r_meas, r_meas]).astype(np.float32)
    kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0
    kf.statePost = np.zeros((6, 1), dtype=np.float32)
    return kf


def kf_set_state_2d(kf: cv2.KalmanFilter, x: float, y: float) -> None:
    kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)


def kf_set_state_3d(kf: cv2.KalmanFilter, x: float, y: float, z: float) -> None:
    kf.statePost = np.array([[x], [y], [z], [0.0], [0.0], [0.0]], dtype=np.float32)


# ====== MAIN ======

def main() -> None:
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None or dist_coeffs is None:
        print("⚠️  Калибровка не найдена: pose/оси не рисуем, только 2D.")
    else:
        print("✅ Калибровка загружена: pose/оси будут рисоваться.")

    detector, dict_id = create_aruco_detector(ARUCO_DICT_NAME)
    print(f"Используем словарь ArUco: {ARUCO_DICT_NAME} (id={dict_id})")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру с id={CAMERA_ID}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30.0
    dt = 1.0 / float(fps)

    HOLD_FRAMES = int(round(HOLD_SECONDS * fps))
    FORGET_FRAMES = int(round(FORGET_SECONDS * fps))

    # Память видимости
    last_seen_frame: Dict[int, int] = {}

    # Последняя ориентация (на случай hold)
    last_rvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    # Kalman на центр (в пикселях) и на tvec (в метрах)
    kf_center: Dict[int, cv2.KalmanFilter] = {}
    kf_tvec: Dict[int, cv2.KalmanFilter] = {}

    frame_idx = 0
    aruco = cv2.aruco

    print("Нажми 'q' в окне видео, чтобы выйти.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Кадр не получен, выходим.")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = detector.detectMarkers(gray)
        seen_ids = set()

        if ids is not None and len(ids) > 0:
            ids = ids.reshape(-1)
            seen_ids = set(int(x) for x in ids)

            aruco.drawDetectedMarkers(frame, corners, ids)

            # --- 2D центры ---
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id)
                last_seen_frame[marker_id] = frame_idx

                pts = corners[i][0]  # (4,2)
                center = pts.mean(axis=0)  # (x,y) float

                if USE_KALMAN_CENTER:
                    if marker_id not in kf_center:
                        kf_center[marker_id] = make_kalman_2d(
                            dt, CENTER_PROCESS_NOISE_POS, CENTER_PROCESS_NOISE_VEL, CENTER_MEAS_NOISE
                        )
                        kf_set_state_2d(kf_center[marker_id], float(center[0]), float(center[1]))

                    kf = kf_center[marker_id]
                    _pred = kf.predict()
                    meas = np.array([[center[0]], [center[1]]], dtype=np.float32)
                    est = kf.correct(meas)  # est = statePost
                    cx, cy = int(est[0, 0]), int(est[1, 0])
                else:
                    cx, cy = int(center[0]), int(center[1])

                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"id={marker_id}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # --- Pose ---
            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
                )

                for marker_id, rvec, tvec in zip(ids, rvecs, tvecs):
                    marker_id = int(marker_id)
                    last_rvecs[marker_id] = rvec

                    t = tvec.reshape(-1).astype(np.float32)  # (3,)

                    if USE_KALMAN_TVEC:
                        if marker_id not in kf_tvec:
                            kf_tvec[marker_id] = make_kalman_3d(
                                dt, TVEC_PROCESS_NOISE_POS, TVEC_PROCESS_NOISE_VEL, TVEC_MEAS_NOISE
                            )
                            kf_set_state_3d(kf_tvec[marker_id], float(t[0]), float(t[1]), float(t[2]))

                        kf = kf_tvec[marker_id]
                        _pred = kf.predict()
                        meas = np.array([[t[0]], [t[1]], [t[2]]], dtype=np.float32)
                        est = kf.correct(meas)
                        t_filtered = np.array([est[0, 0], est[1, 0], est[2, 0]], dtype=np.float32)
                    else:
                        t_filtered = t

                    draw_axes(frame, camera_matrix, dist_coeffs, rvec, t_filtered, MARKER_LENGTH_METERS * 0.5)

                    dist_m = float(np.linalg.norm(t_filtered))
                    print(f"id={marker_id}: distance ≈ {dist_m:.3f} m", end="\r")

        # --- HOLD: дорисовываем пропавшие маркеры по предсказанию Kalman ---
        to_delete = []
        for marker_id, last_f in list(last_seen_frame.items()):
            age = frame_idx - last_f

            if age > FORGET_FRAMES:
                to_delete.append(marker_id)
                continue

            if marker_id in seen_ids:
                continue

            if age <= HOLD_FRAMES:
                # 2D предсказание
                if USE_KALMAN_CENTER and marker_id in kf_center:
                    est = kf_center[marker_id].predict()
                    cx, cy = int(est[0, 0]), int(est[1, 0])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(frame, f"id={marker_id} (held)", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # 3D предсказание + last_rvec
                if (camera_matrix is not None and dist_coeffs is not None and
                        USE_KALMAN_TVEC and marker_id in kf_tvec):
                    rvec = last_rvecs.get(marker_id)
                    if rvec is not None:
                        est = kf_tvec[marker_id].predict()
                        t_pred = np.array([est[0, 0], est[1, 0], est[2, 0]], dtype=np.float32)
                        draw_axes(frame, camera_matrix, dist_coeffs, rvec, t_pred, MARKER_LENGTH_METERS * 0.5)

        # чистим старые
        for marker_id in to_delete:
            last_seen_frame.pop(marker_id, None)
            last_rvecs.pop(marker_id, None)
            kf_center.pop(marker_id, None)
            kf_tvec.pop(marker_id, None)

        cv2.imshow("ArUco demo (Kalman)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
