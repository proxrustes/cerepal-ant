
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


CAMERA_MATRIX_PATH = Path("camera_matrix.npy")
DIST_COEFFS_PATH = Path("dist_coeffs.npy")

# ====== OpenCV helpers ======

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
    pts = corner_block[0] if corner_block.ndim == 3 else corner_block
    pts = pts.astype(np.float32)
    return float(cv2.contourArea(pts))


def project_point(camera_matrix, dist_coeffs, p_cam: np.ndarray) -> Tuple[int, int]:
    """p_cam shape (3,) in camera frame. Returns pixel (u,v)."""
    p = np.asarray(p_cam, dtype=np.float64).reshape(1, 1, 3)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    imgpts, _ = cv2.projectPoints(p, rvec, tvec, camera_matrix, dist_coeffs)
    u, v = imgpts.reshape(2)
    return int(round(u)), int(round(v))


def draw_cube_axes_on_image(img, camera_matrix, dist_coeffs, T_C_from_K: np.ndarray, axis_len: float) -> None:
    """
    Рисуем оси куба (в картинке), используя T_C<-K.
    axis_len в метрах.
    """
    R = T_C_from_K[:3, :3]
    t = T_C_from_K[:3, 3]

    def to_cam(pK):
        pK = np.asarray(pK, dtype=np.float64)
        return (R @ pK) + t

    origin = to_cam((0, 0, 0))
    px = to_cam((axis_len, 0, 0))
    py = to_cam((0, axis_len, 0))
    pz = to_cam((0, 0, axis_len))

    o = project_point(camera_matrix, dist_coeffs, origin)
    x = project_point(camera_matrix, dist_coeffs, px)
    y = project_point(camera_matrix, dist_coeffs, py)
    z = project_point(camera_matrix, dist_coeffs, pz)

    cv2.line(img, o, x, (0, 0, 255), 2)   # X (красный)
    cv2.line(img, o, y, (0, 255, 0), 2)   # Y (зелёный)
    cv2.line(img, o, z, (255, 0, 0), 2)   # Z (синий)

    cv2.putText(img, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
