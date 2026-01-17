
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

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
    """
    r20 = float(R[2, 0])
    r20 = max(-1.0, min(1.0, r20))

    pitch = -np.arcsin(r20)
    cp = np.cos(pitch)

    if abs(cp) < 1e-6:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0.0
    else:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)

    return (float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll)))



def ema_vec(prev: Optional[np.ndarray], new: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return new
    return alpha * new + (1.0 - alpha) * prev


def motion_label(v: np.ndarray, deadzone: float) -> Tuple[str, float]:
    """
    v в м/с в координатах куба.
    Возвращает (текст направления, скорость_мс).
    """
    speed = float(np.linalg.norm(v))
    if speed < deadzone:
        return "stopped", speed

    ax = int(np.argmax(np.abs(v)))
    s = float(v[ax])

    if ax == 0:  # X
        return ("right" if s > 0 else "left"), speed
    if ax == 1:  # Y
        return ("up" if s > 0 else "down"), speed
    # ax == 2: Z
    return ("toward_front" if s > 0 else "toward_back"), speed


def direction_with_hysteresis(
    v: np.ndarray,
    deadzone: float,
    last_dir: str,
    axis_switch_ratio: float,
) -> str:
    speed = float(np.linalg.norm(v))
    if speed < deadzone:
        return "stopped"

    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    ax = np.abs([vx, vy, vz])
    k = int(np.argmax(ax))

    # кандидат
    if k == 0:
        cand = "right" if vx > 0 else "left"
        primary = ax[0]; secondary = max(ax[1], ax[2])
    elif k == 1:
        cand = "up" if vy > 0 else "down"
        primary = ax[1]; secondary = max(ax[0], ax[2])
    else:
        cand = "toward_front" if vz > 0 else "toward_back"
        primary = ax[2]; secondary = max(ax[0], ax[1])

    # если прошлое направление по другой оси — не переключаемся,
    # пока преимущество не станет "явным"
    if last_dir != "stopped":
        last_axis = 0 if last_dir in ("right", "left") else 1 if last_dir in ("up", "down") else 2
        if last_axis != k:
            # новая ось должна быть явно больше второй (иначе это диагональ/шум)
            if secondary > 1e-9 and (primary / secondary) < axis_switch_ratio:
                return last_dir

    return cand



def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to quaternion (w,x,y,z).
    Assumes R is a proper rotation matrix.
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)

    if t > 0.0:
        S = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    else:
        # pick the largest diagonal element
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    # normalize (important for numeric stability)
    q /= np.linalg.norm(q)
    # optional: enforce w>=0 to avoid sign flips between frames
    if q[0] < 0:
        q = -q
    return q