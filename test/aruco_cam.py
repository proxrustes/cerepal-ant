#!/usr/bin/env python3
"""
ArUco demo + EMA + hold + pose robot-in-cube + JSONL stream + SNAPSHOT.

- Камера на роботе
- Куб стоит в комнате
- На 5 гранях куба теги с id: 1..5
- По видимому тегу считаем позу камеры в системе координат куба
- Печатаем JSON lines (timestamped) в stdout
- Нажми 'p' чтобы сделать снапшот с разметкой и расчётами
"""

from __future__ import annotations

import collections
from typing import Dict, Optional

import cv2
import numpy as np

import time
import json
from datetime import datetime

from lib.cube_params import tag_pose_on_face
from lib.draw_helpers import draw_text_panel
from lib.opecv_helpers import create_aruco_detector, draw_axes, draw_cube_axes_on_image, ema, load_camera_params, marker_area, project_point
from lib.transform_helpers import ema_vec, invert_T, motion_label, rot_to_ypr_deg, rt_to_T


# ====== НАСТРОЙКИ ======

ARUCO_DICT_NAME = "DICT_4X4_1000"
CAMERA_ID = 0

# ФИЗИКА (твои данные)
CUBE_L = 0.04343                 # длина ребра куба (м)
MARKER_LENGTH_METERS = 0.03797   # длина стороны маркера (м) <-- важно!

SMOOTHING_ALPHA = 0.3

HOLD_SECONDS = 2.0
FORGET_SECONDS = 5.0

# ====== MOTION (куда едет робот) ======
prev_pos = None          # np.array shape (3,)
prev_ts = None           # time.time()
vel_ema = None           # сглаженная скорость (3,)
VEL_ALPHA = 0.4          # 0..1, больше = резче
VEL_DEADZONE = 0.02      # м/с, ниже считаем "стоит"

HALF = CUBE_L / 2.0

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

# ====== НАПРАВЛЕНИЯ В КООРДИНАТАХ КУБА (K-frame) ======
UP_TO_CEILING = (0, +1, 0)   # +Y
UP_TO_FLOOR   = (0, -1, 0)   # -Y
TOWARD_FRONT  = (0, 0, +1)   # +Z
TOWARD_BACK   = (0, 0, -1)   # -Z
TOWARD_RIGHT  = (+1, 0, 0)   # +X
TOWARD_LEFT   = (-1, 0, 0)   # -X

# ТВОИ текущие настройки (можешь крутить)
TAG_UP = {
    2: UP_TO_CEILING,
    3: TOWARD_BACK,
    4: UP_TO_FLOOR,
    1: TOWARD_BACK,
    5: TOWARD_BACK,
}

TAG_YAW_DEG = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

TAG_IN_CUBE: Dict[int, np.ndarray] = {
    tid: tag_pose_on_face(
        face_normal=np.array(FACE_NORMAL[tid], dtype=np.float64),
        face_up=np.array(TAG_UP[tid], dtype=np.float64),
        face_center=np.array(FACE_CENTER[tid], dtype=np.float64),
        yaw_deg=float(TAG_YAW_DEG[tid]),
    )
    for tid in FACE_NORMAL.keys()
}


# ====== MAIN ======

def main() -> None:
    # ====== MOTION (куда едет робот) ======
    prev_pos: Optional[np.ndarray] = None
    prev_ts: Optional[float] = None
    vel_ema: Optional[np.ndarray] = None

    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None or dist_coeffs is None:
        print("⚠️  Калибровка не найдена: pose не будет считаться.")
    else:
        print("✅ Калибровка загружена: pose будет считаться.")

    detector, dict_id = create_aruco_detector(ARUCO_DICT_NAME)
    print(f"Используем словарь ArUco: {ARUCO_DICT_NAME} (id={dict_id})")
    print(f"CUBE_L={CUBE_L} m, MARKER_LENGTH_METERS={MARKER_LENGTH_METERS} m")
    print("Нажми 'q' чтобы выйти. Нажми 'p' чтобы сделать снимок с расчётами.")

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

    # last pose camera-in-cube per tag
    last_TK_from_C: Dict[int, Optional[np.ndarray]] = collections.defaultdict(lambda: None)

    aruco = cv2.aruco
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Кадр не получен, выходим.")
            break

        frame_idx += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _rejected = detector.detectMarkers(gray)

        seen_ids = set()
        areas: Dict[int, float] = {}
        # Для снапшота: сохраняем метрики текущего кадра (только видимые)
        frame_metrics = {}  # id -> dict (tvec_cam, dist_cam_tag, dist_cam_cube, dist_cam_face, T_C_from_K)

        if ids is not None and len(ids) > 0:
            ids = ids.reshape(-1)
            seen_ids = set(int(x) for x in ids)

            aruco.drawDetectedMarkers(frame, corners, ids)

            for i, marker_id in enumerate(ids):
                marker_id = int(marker_id)
                last_seen_frame[marker_id] = frame_idx

                pts = corners[i][0]
                center = pts.mean(axis=0)

                smoothed = ema(smoothed_centers[marker_id], center, SMOOTHING_ALPHA)
                smoothed_centers[marker_id] = smoothed

                cx, cy = int(smoothed[0]), int(smoothed[1])
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"id={marker_id}", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                areas[marker_id] = marker_area(corners[i])

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

                    draw_axes(frame, camera_matrix, dist_coeffs, rvec, smoothed_t, MARKER_LENGTH_METERS * 0.5)

                    if marker_id in TAG_IN_CUBE:
                        T_C_from_T = rt_to_T(rvec, smoothed_t)          # tag -> camera
                        T_K_from_T = TAG_IN_CUBE[marker_id]             # tag -> cube
                        T_C_from_K = T_C_from_T @ invert_T(T_K_from_T)  # cube -> camera
                        T_K_from_C = invert_T(T_C_from_K)               # camera -> cube

                        last_TK_from_C[marker_id] = T_K_from_C

                        # Метрики (в координатах камеры)
                        pC_cube_center = T_C_from_K[:3, 3]  # куб-центр в камере
                        dist_cam_cube = float(np.linalg.norm(pC_cube_center))

                        dist_cam_tag = float(np.linalg.norm(smoothed_t))

                        # центр конкретной грани (в кубе) -> в камеру
                        pK_face = np.array(FACE_CENTER[marker_id], dtype=np.float64)
                        pC_face = (T_C_from_K[:3, :3] @ pK_face) + T_C_from_K[:3, 3]
                        dist_cam_face = float(np.linalg.norm(pC_face))

                        frame_metrics[marker_id] = {
                            "dist_cam_tag": dist_cam_tag,
                            "dist_cam_cube": dist_cam_cube,
                            "dist_cam_face": dist_cam_face,
                            "pC_cube_center": pC_cube_center,
                            "T_C_from_K": T_C_from_K,
                        }

        # HOLD отрисовка (как у тебя)
        to_delete = []
        for marker_id, last_f in list(last_seen_frame.items()):
            age = frame_idx - last_f

            if age > FORGET_FRAMES:
                to_delete.append(marker_id)
                continue

            if marker_id in seen_ids:
                continue

            if age <= HOLD_FRAMES:
                c = smoothed_centers.get(marker_id)
                if c is not None:
                    cx, cy = int(c[0]), int(c[1])
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                    cv2.putText(frame, f"id={marker_id} (held)", (cx + 5, cy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                if camera_matrix is not None and dist_coeffs is not None:
                    rvec = last_rvecs.get(marker_id)
                    tvec = last_tvecs.get(marker_id)
                    if rvec is not None and tvec is not None:
                        draw_axes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_LENGTH_METERS * 0.5)

        for marker_id in to_delete:
            last_seen_frame.pop(marker_id, None)
            smoothed_centers.pop(marker_id, None)
            smoothed_tvecs.pop(marker_id, None)
            last_rvecs.pop(marker_id, None)
            last_tvecs.pop(marker_id, None)
            last_TK_from_C.pop(marker_id, None)
            areas.pop(marker_id, None)

        # Выбор "лучшего" тега (как раньше) и JSONL stream (оставляем)
        chosen_id = None
        chosen_T = None
        chosen_held = False

        # кандидаты только видимые (для стрима)
        visible_candidate_ids = [mid for mid in seen_ids if last_TK_from_C.get(mid) is not None]
        if len(visible_candidate_ids) > 0:
            chosen_id = max(visible_candidate_ids, key=lambda mid: areas.get(mid, 0.0))
            chosen_T = last_TK_from_C[chosen_id]
            chosen_held = False

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

        # ====== LIVE OVERLAY ======
        if len(frame_metrics) > 0:
            visible_list = sorted(list(frame_metrics.keys()))
            best_id = max(frame_metrics.keys(), key=lambda mid: areas.get(mid, 0.0))
            best = frame_metrics[best_id]
            dist_cam_cube = float(best["dist_cam_cube"])

            lines = []
            # верхняя строка (поза)
            chosen_T = last_TK_from_C.get(best_id)
            if chosen_T is not None:
                pos = chosen_T[:3, 3]
                lines.append(f"K<-C  x={pos[0]:+0.3f}  y={pos[1]:+0.3f}  z={pos[2]:+0.3f}   tag={best_id}")

                # motion
                now_ts = time.time()
                cur_pos = pos.astype(np.float64)
                if prev_pos is not None and prev_ts is not None:
                    dt = now_ts - prev_ts
                    if dt > 1e-3:
                        v = (cur_pos - prev_pos) / dt
                        vel_ema = ema_vec(vel_ema, v, VEL_ALPHA)

                prev_pos = cur_pos
                prev_ts = now_ts

                if vel_ema is not None:
                    label, speed = motion_label(vel_ema, VEL_DEADZONE)
                    lines.append(f"motion: {label}   v={speed:.3f} m/s")

            lines.append(f"Visible tags: {visible_list}")
            lines.append(f"Best tag (for cube pose): {best_id}")
            lines.append(f"cam->CUBE_CENTER = {dist_cam_cube:.3f} m")

            for mid in visible_list:
                m = frame_metrics[mid]
                lines.append(
                    f"id={mid}  cam->tag={m['dist_cam_tag']:.3f}m  cam->cube={m['dist_cam_cube']:.3f}m  cam->face={m['dist_cam_face']:.3f}m"
                )

            # рисуем красивую панель слева сверху
            draw_text_panel(
                frame,
                lines,
                origin=(10, 10),
                font_scale=0.75,
                thickness=2,
                bg_alpha=0.55,
                padding=10,
                line_gap=6,
                max_width=frame.shape[1] - 20,
            )

            visible_list = sorted(list(frame_metrics.keys()))

            # best tag: по площади в кадре
            best_id = max(frame_metrics.keys(), key=lambda mid: areas.get(mid, 0.0))
            best = frame_metrics[best_id]

            # расстояние до центра куба (в метрах)
            dist_cam_cube = float(best["dist_cam_cube"])

            # верхняя строка: поза камеры в кубе (если есть)
            chosen_T = last_TK_from_C.get(best_id)

        cv2.imshow("ArUco demo (cube pose)", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # ====== SNAPSHOT по клавише 'p' ======
        if key == ord("p"):
            if camera_matrix is None or dist_coeffs is None:
                print("SNAPSHOT: нет калибровки — не могу посчитать 3D.")
                continue
            if len(frame_metrics) == 0:
                print("SNAPSHOT: нет видимых тегов с позой.")
                continue

            snap = frame.copy()

            # --- 1) собрать центры куба по всем тегам ---
            centers = []  # list of (id, pC_center(3,))
            for mid, m in frame_metrics.items():
                pC = m["pC_cube_center"].reshape(3,)
                centers.append((mid, pC))

            # --- 2) нарисовать каждый центр отдельно ---
            for mid, pC in centers:
                u, v = project_point(camera_matrix, dist_coeffs, pC)
                cv2.circle(snap, (u, v), 5, (0, 255, 255), -1)  # жёлтая точка
                cv2.putText(
                    snap, f"C(mid={mid})", (u + 6, v - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
                )

            # --- 3) простое отбрасывание выброса ---
            # считаем "медианный" центр (по компонентам) и выкидываем те,
            # кто дальше 3 см от медианы
            P = np.stack([p for _mid, p in centers], axis=0)  # (N,3)
            med = np.median(P, axis=0)
            dists = np.linalg.norm(P - med[None, :], axis=1)
            keep = dists < 0.03  # 3 см порог (можно сделать 0.02)

            if np.any(keep):
                Pk = P[keep]
                mean_center = np.mean(Pk, axis=0)
            else:
                mean_center = np.mean(P, axis=0)  # если все "плохие", хоть что-то

            # --- 4) нарисовать итоговый центр (mean) ---
            u, v = project_point(camera_matrix, dist_coeffs, mean_center)
            cv2.circle(snap, (u, v), 7, (255, 255, 255), -1)
            cv2.putText(
                snap, "CUBE CENTER (mean)", (u + 10, v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )

            # --- 5) выбрать лучший тег ДЛЯ ОРИЕНТАЦИИ осей, но центр взять mean ---
            best_id = max(frame_metrics.keys(), key=lambda mid: areas.get(mid, 0.0))
            T_C_from_K = frame_metrics[best_id]["T_C_from_K"].copy()
            T_C_from_K[:3, 3] = mean_center  # заменили только translation на mean

            draw_cube_axes_on_image(snap, camera_matrix, dist_coeffs, T_C_from_K, axis_len=CUBE_L * 0.75)

            # --- 6) печать репорта (очень полезно для твиков) ---
            print("\n=== SNAPSHOT REPORT ===")
            print("visible_ids:", sorted(list(frame_metrics.keys())))
            print("best_tag_for_axes:", best_id)

            for (mid, pC), dist in zip(centers, dists):
                print(f"center from tag {mid}: |pC|={np.linalg.norm(pC):.4f} m, "
                      f"delta_to_median={dist*100:.1f} cm")

            print(f"mean_center |pC|={np.linalg.norm(mean_center):.4f} m")
            print("=======================\n")

            # --- 7) сохранить ---
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"cube_snapshot_{ts}.png"
            cv2.imwrite(out_name, snap)
            print("saved:", out_name)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
