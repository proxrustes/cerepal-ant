from collections import deque
from dataclasses import dataclass, field
import time
from typing import Deque, Optional, Tuple

import numpy as np

from lib.transform_helpers import ema_vec


@dataclass
class MotionEstimator:
    # --- filtering ---
    pos_alpha: float = 0.12      # EMA по позиции (0..1). меньше = плавнее
    vel_alpha: float = 0.15      # EMA по скорости (0..1)
    deadzone: float = 0.03       # м/с

    # --- velocity from window ---
    window_sec: float = 0.70     # сколько секунд держим точек для оценки скорости
    min_points: int = 10

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

