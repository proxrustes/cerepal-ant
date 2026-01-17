#!/usr/bin/env python3
"""
ArUco demo + EMA + hold + pose robot-in-cube + JSONL stream.

- Камера на роботе
- Куб стоит в комнате
- На 5 гранях куба теги с id: 1..5 (как ты описала)
- По видимому тегу считаем позу камеры в системе координат куба
- Печатаем JSON lines (timestamped) в stdout
"""

from __future__ import annotations

import collections
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import time
import json


# ====== НАСТРОЙКИ ======

ARUCO_DICT_NAME = "DICT_4X4_1000"
CAMERA_ID = 0

# ФИЗИКА (из твоих данных)
CUBE_L = 0.04343                 # длина ребра куба (м)
MARKER_LENGTH_METERS = 0.03797   # длина стороны маркера (м) <-- важно!

CAMERA_MATRIX_PATH = Path("camera_matrix.npy")
DIST_COEFFS_PATH = Path("dist_coeffs.npy")

SMOOTHING_ALPHA = 0.3

HOLD_SECONDS = 2.0
FORGET_SECONDS = 5.0


# ====== МАТЕМАТИКА ТРАНСФОРМОВ ======

def rt_to_T(rvec, tvec) -> np.ndarray:
    """OpenCV pose -> 4x4, возвращает T_C<-T (tag -> camera)."""
    R, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
    t = np.asarray(tvec).reshape(3, 1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Инверсия 4x4 SE(3)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_to_ypr_deg(R: np.ndarray) -> Tuple[float, float, float]:
    """
    yaw/pitch/roll в градусах из матрицы R.
    Конвенция: R = Rz(yaw) * Ry(pitch) * Rx(roll).
    yaw вокруг Z, pitch вокруг Y, roll вокруг X.
    """
    # защита от численных глюков
    r20 = float(R[2, 0])
    r20 = max(-1.0, min(1.0, r20))

    pitch = -np.arcsin(r20)
    cp = np.cos(pitch)

    if abs(cp) < 1e-6:
        # gimbal lock
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)

    return (float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll)))


# ====== КУБ: где сидит каждый тег (T_K<-T_id) + твики ориентации ======

def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero vector")
    return v / n


def rot_axis_angle(axis, deg):
    axis = norm(axis)
    rvec = axis * np.deg2rad(deg)
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def make_T_from_axes_and_pos(x_axis, y_axis, z_axis, pos):
    # R columns = axes of tag frame expressed in cube frame
    R = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T  # T_K<-T


def tag_pose_on_face(face_normal, face_up, face_center, yaw_deg=0.0):
    """
    Строим T_K<-T (tag->cube) по трем понятным штукам:
      - face_normal: куда "наружу" смотрит грань (в координатах куба)
      - face_up: куда смотрит "верх распечатки" тега (в координатах куба)
      - yaw_deg: докрутка в плоскости тега на 0/90/180/270 (если наклеено криво)

    ВАЖНО: у OpenCV ArUco ось +Y тега направлена ВНИЗ по картинке,
    поэтому y_tag в 3D направляем ПРОТИВ face_up (ставим минус).
    """
    z = norm(face_normal)          # z_tag "наружу"
    y = -norm(face_up)             # y_tag "вниз по картинке"
    x = np.cross(y, z)
    x = norm(x)
    y = np.cross(z, x)             # переортонормируем

    if abs(yaw_deg) > 1e-9:
        Rz = rot_axis_angle(z, yaw_deg)
        x = Rz @ x
        y = Rz @ y

    return make_T_from_axes_and_pos(x, y, z, face_center)


HALF = CUBE_L / 2.0

# ТВОЙ mapping id -> грань
FACE_NORMAL = {
    2: (0, 0, +1),   # +Z front
    3: (+1, 0, 0),   # +X right
    4: (0, 0, -1),   # -Z back
    1: (-1, 0, 0),   # -X left
    5: (0, +1, 0),   # +Y top
}

FACE_CENTER = {
    2: (0, 0, +HALF),
    3: (+HALF, 0, 0),
    4: (0, 0, -HALF),
    1: (-HALF, 0, 0),
    5: (0, +HALF, 0),
}

# ====== ТВИКИ ОРИЕНТАЦИИ ======
# 1) TAG_UP: куда смотрит "верх распечатки" для каждой грани (в координатах куба).

# ====== НАПРАВЛЕНИЯ В КООРДИНАТАХ КУБА (K-frame) ======
# Чтобы не писать цифры руками.

UP_TO_CEILING = (0, +1, 0)   # к потолку (+Y)
UP_TO_FLOOR   = (0, -1, 0)   # к полу    (-Y)

TOWARD_FRONT  = (0, 0, +1)   # к передней (+Z)
TOWARD_BACK   = (0, 0, -1)   # к задней   (-Z)

TOWARD_RIGHT  = (+1, 0, 0)   # к правой   (+X)
TOWARD_LEFT   = (-1, 0, 0)   # к левой    (-X)


TAG_UP = {
    2: UP_TO_CEILING,
    3: TOWARD_BACK,
    4: UP_TO_FLOOR,
    1: TOWARD_BACK,
    5: TOWARD_BACK,
}

# 2) TAG_YAW_DEG: если конкретный тег наклеен повернутым (0/90/180/270).
TAG_YAW_DEG = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

# Итог: T_K<-T_id (tag->cube)
TAG_IN_CUBE: Dict[int, np.ndarray] = {
    tid: tag_pose_on_face(
        face_normal=np.array(FACE_NORMAL[tid], dtype=np.float64),
        face_up=np.array(TAG_UP[tid], dtype=np.float64),
        face_center=np.array(FACE_CENTER[tid], dtype=np.float64),
        yaw_deg=float(TAG_YAW_DEG[tid]),
    )
    for tid in FACE_NORMAL.keys()
}


# ====== ВСПОМОГАТЕЛЬНЫЕ (OpenCV) ======

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


def ema(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


def marker_area(corner_block: np.ndarray) -> float:
    """corner_block shape (1,4,2) or (4,2)."""
    pts = corner_block[0] if corner_block.ndim == 3 else corner_block
    pts = pts.astype(np.float32)
    return float(cv2.contourArea(pts))


# ====== MAIN ======

def main() -> None:
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None or dist_coeffs is None:
        print("⚠️  Калибровка не найдена: pose не будет считаться.")
    else:
        print("✅ Калибровка загружена: pose будет считаться.")

    detector, dict_id = create_aruco_detector(ARUCO_DICT_NAME)
    print(f"Используем словарь ArUco: {ARUCO_DICT_NAME} (id={dict_id})")
    print(f"CUBE_L={CUBE_L} m, MARKER_LENGTH_METERS={MARKER_LENGTH_METERS} m")

    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру с id={CAMERA_ID}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps:
        fps = 30.0

    HOLD_FRAMES = int(round(HOLD_SECONDS * fps))
    FORGET_FRAMES = int(round(FORGET_SECONDS * fps))

    smoothed_centers: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)
    smoothed_tvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    last_seen_frame: Dict[int, int] = {}
    last_rvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)
    last_tvecs: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    # Храним последнюю позу камеры в координатах куба (T_K<-C) по каждому тегу
    last_TK_from_C: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    aruco = cv2.aruco
    frame_idx = 0

    print("Нажми 'q' в окне видео, чтобы выйти.")
    print("JSONL stream идёт в stdout (по 1 строке на кадр, где есть поза).")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Кадр не получен, выходим.")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = detector.detectMarkers(gray)

        seen_ids = set()
        areas: Dict[int, float] = {}   # площадь маркера в пикселях (для выбора "лучшего")

        # ====== 1) обычная отрисовка + обновление last_* ======
        if ids is not None and len(ids) > 0:
            ids = ids.reshape(-1)
            seen_ids = set(int(x) for x in ids)

            aruco.drawDetectedMarkers(frame, corners, ids)

            # центры (EMA)
            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id)
                last_seen_frame[marker_id] = frame_idx

                pts = corners[i][0]  # (4,2)
                center = pts.mean(axis=0)

                smoothed = ema(smoothed_centers[marker_id], center, SMOOTHING_ALPHA)
                smoothed_centers[marker_id] = smoothed

                cx, cy = int(smoothed[0]), int(smoothed[1])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"id={marker_id}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                areas[marker_id] = marker_area(corners[i])

            # pose + вычисление robot-in-cube
            if camera_matrix is not None and dist_coeffs is not None:
                rvecs, tvecs, _obj_points = aruco.estimatePoseSingleMarkers(
                    corners, MARKER_LENGTH_METERS, camera_matrix, dist_coeffs
                )

                for marker_id, rvec, tvec in zip(ids, rvecs, tvecs):
                    marker_id = int(marker_id)

                    tvec = tvec.reshape(-1)
                    smoothed_t = ema(smoothed_tvecs[marker_id], tvec, SMOOTHING_ALPHA)
                    smoothed_tvecs[marker_id] = smoothed_t

                    last_rvecs[marker_id] = rvec
                    last_tvecs[marker_id] = smoothed_t

                    # Рисуем оси тега в координатах камеры (визуально полезно)
                    draw_axes(frame, camera_matrix, dist_coeffs, rvec, smoothed_t, MARKER_LENGTH_METERS * 0.5)

                    # ====== Ключевое: считаем T_K<-C (робот в координатах куба) ======
                    if marker_id in TAG_IN_CUBE:
                        T_C_from_T = rt_to_T(rvec, smoothed_t)         # tag -> camera
                        T_K_from_T = TAG_IN_CUBE[marker_id]            # tag -> cube
                        T_C_from_K = T_C_from_T @ invert_T(T_K_from_T) # cube -> camera
                        T_K_from_C = invert_T(T_C_from_K)              # camera -> cube (то, что надо)

                        last_TK_from_C[marker_id] = T_K_from_C

        # ====== 2) HOLD отрисовка + выбор кандидатов на позу ======
        candidates: Dict[int, Tuple[np.ndarray, bool]] = {}  # id -> (T_K<-C, held?)

        # видимые теги (если pose был посчитан)
        for mid in list(seen_ids):
            T = last_TK_from_C.get(mid)
            if T is not None:
                candidates[mid] = (T, False)

        # held теги (если недавно пропали)
        to_delete = []
        for marker_id, last_f in list(last_seen_frame.items()):
            age = frame_idx - last_f

            if age > FORGET_FRAMES:
                to_delete.append(marker_id)
                continue

            if marker_id in seen_ids:
                continue

            if age <= HOLD_FRAMES:
                # рисуем held центр
                c = smoothed_centers.get(marker_id)
                if c is not None:
                    cx, cy = int(c[0]), int(c[1])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(frame, f"id={marker_id} (held)", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # и held оси (тега в камере)
                if camera_matrix is not None and dist_coeffs is not None:
                    rvec = last_rvecs.get(marker_id)
                    tvec = last_tvecs.get(marker_id)
                    if rvec is not None and tvec is not None:
                        draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS * 0.5)

                # если есть последняя T_K<-C — добавляем как held-кандидата
                T = last_TK_from_C.get(marker_id)
                if T is not None and marker_id not in candidates:
                    candidates[marker_id] = (T, True)

        # чистим старые
        for marker_id in to_delete:
            last_seen_frame.pop(marker_id, None)
            smoothed_centers.pop(marker_id, None)
            smoothed_tvecs.pop(marker_id, None)
            last_rvecs.pop(marker_id, None)
            last_tvecs.pop(marker_id, None)
            last_TK_from_C.pop(marker_id, None)
            areas.pop(marker_id, None)

        # ====== 3) Выбираем "лучший" тег для итоговой позы кадра ======
        # Логика:
        # - если есть видимые кандидаты: выбираем по максимальной площади в кадре (точнее)
        # - иначе: берём любой held (например с наименьшим age, но упростим)
        chosen_id = None
        chosen_T = None
        chosen_held = False

        visible_candidate_ids = [mid for mid, (_T, held) in candidates.items() if not held]
        if len(visible_candidate_ids) > 0:
            chosen_id = max(visible_candidate_ids, key=lambda mid: areas.get(mid, 0.0))
            chosen_T, chosen_held = candidates[chosen_id]
        elif len(candidates) > 0:
            # берём held, который пропал "наименее давно"
            chosen_id = min(candidates.keys(), key=lambda mid: frame_idx - last_seen_frame.get(mid, frame_idx))
            chosen_T, chosen_held = candidates[chosen_id]

        # ====== 4) STREAM: печатаем позу робота относительно куба ======
        if chosen_id is not None and chosen_T is not None:
            pos = chosen_T[:3, 3]
            R = chosen_T[:3, :3]
            yaw, pitch, roll = rot_to_ypr_deg(R)

            msg = {
                "ts_ns": time.time_ns(),
                "tag_used": int(chosen_id),
                "held": bool(chosen_held),
                "pos_m": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
                "ypr_deg": {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)},
            }
            print(json.dumps(msg), flush=True)

            # маленький оверлей на кадр
            overlay = f"K<-C  x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}  tag={chosen_id}{' held' if chosen_held else ''}"
            cv2.putText(frame, overlay, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ArUco demo (cube pose)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
