
from typing import Optional, Tuple

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

