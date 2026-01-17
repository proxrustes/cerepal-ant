
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


def motion_label_stable(
    v: np.ndarray,
    deadzone: float,
    dominance: float = 0.60,   # 0.6 = ось должна быть заметно доминирующей
) -> Tuple[str, float]:
    """
    Return (label, speed). If motion is diagonal and no axis dominates -> "diagonal".
    """
    speed = float(np.linalg.norm(v))
    if speed < deadzone:
        return "stopped", speed

    av = np.abs(v)
    ssum = float(av.sum()) + 1e-12
    ax = int(np.argmax(av))
    frac = float(av[ax] / ssum)  # насколько доминирует ось

    if frac < dominance:
        return "diagonal", speed

    if ax == 0:  # X
        return ("right" if v[0] > 0 else "left"), speed
    if ax == 1:  # Y
        return ("up" if v[1] > 0 else "down"), speed
    return ("toward_front" if v[2] > 0 else "toward_back"), speed

@dataclass
class MotionEstimator:
    # --- filtering ---
    pos_alpha: float = 0.25      # EMA по позиции (0..1). меньше = плавнее
    vel_alpha: float = 0.35      # EMA по скорости (0..1)
    deadzone: float = 0.02       # м/с

    # --- velocity from window ---
    window_sec: float = 0.35     # сколько секунд держим точек для оценки скорости
    min_points: int = 5

    # --- label stability ---
    dominance: float = 0.60
    stable_frames: int = 4       # новый label должен держаться N кадров прежде чем принять
    max_jump_m: float = 0.10     # если позиция прыгнула >10см за кадр -> считаем это глитчем (например смена тега)

    # --- state ---
    pos_ema: Optional[np.ndarray] = None
    vel_ema: Optional[np.ndarray] = None
    buf: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=200))

    last_label: str = "stopped"
    pending_label: Optional[str] = None
    pending_count: int = 0

    last_pos_raw: Optional[np.ndarray] = None
    last_ts: Optional[float] = None

    def reset(self) -> None:
        self.pos_ema = None
        self.vel_ema = None
        self.buf.clear()
        self.last_label = "stopped"
        self.pending_label = None
        self.pending_count = 0
        self.last_pos_raw = None
        self.last_ts = None

    def update(self, pos: np.ndarray, ts: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, str, float]:
        """
        Input:
          pos: (3,) position in cube frame
          ts: seconds, default time.time()

        Output:
          (pos_smooth, vel_smooth, stable_label, speed)
        """
        if ts is None:
            ts = time.time()

        pos = np.asarray(pos, dtype=np.float64).reshape(3,)

        # --- detect jumps (often from tag switching / bad pose) ---
        if self.last_pos_raw is not None:
            jump = float(np.linalg.norm(pos - self.last_pos_raw))
            if jump > self.max_jump_m:
                # не “обнуляем” всё, но не используем этот рывок для скорости
                # вариант мягче: просто сбросить буфер скорости
                self.buf.clear()
        self.last_pos_raw = pos

        # --- smooth position first ---
        self.pos_ema = ema_vec(self.pos_ema, pos, self.pos_alpha)

        # --- maintain buffer for velocity regression ---
        self.buf.append((ts, self.pos_ema.copy()))

        # remove old points beyond window_sec
        while len(self.buf) >= 2 and (ts - self.buf[0][0]) > self.window_sec:
            self.buf.popleft()

        # --- estimate velocity by least squares over window ---
        v_est = np.zeros(3, dtype=np.float64)
        if len(self.buf) >= self.min_points:
            t0 = self.buf[0][0]
            T = np.array([t - t0 for (t, _p) in self.buf], dtype=np.float64)  # (N,)
            P = np.stack([p for (_t, p) in self.buf], axis=0)                 # (N,3)
            # slope = cov(T,P)/var(T)
            varT = float(np.dot(T - T.mean(), T - T.mean())) + 1e-12
            Tc = (T - T.mean()).reshape(-1, 1)                                # (N,1)
            Pc = (P - P.mean(axis=0)).astype(np.float64)                      # (N,3)
            slope = (Tc.T @ Pc).reshape(3,) / varT                            # (3,)
            v_est = slope

        # --- smooth velocity ---
        if len(self.buf) >= self.min_points:
            self.vel_ema = ema_vec(self.vel_ema, v_est, self.vel_alpha)
        else:
            self.vel_ema = v_est if self.vel_ema is None else self.vel_ema


        # --- label with dominance + hysteresis ---
        label_now, speed = motion_label_stable(self.vel_ema, self.deadzone, dominance=self.dominance)

        if label_now == self.last_label:
            self.pending_label = None
            self.pending_count = 0
        else:
            if self.pending_label != label_now:
                self.pending_label = label_now
                self.pending_count = 1
            else:
                self.pending_count += 1
                if self.pending_count >= self.stable_frames:
                    self.last_label = label_now
                    self.pending_label = None
                    self.pending_count = 0

        return self.pos_ema, self.vel_ema, self.last_label, float(speed)
    


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