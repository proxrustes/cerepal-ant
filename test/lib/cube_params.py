
import cv2
import numpy as np


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
    Строим T_K<-T (tag->cube):
      - face_normal: наружу грани (в координатах куба)
      - face_up: куда смотрит верх распечатки (в координатах куба)
      - yaw_deg: докрутка тега на 0/90/180/270 в плоскости грани

    Важно: у OpenCV ArUco ось +Y тега направлена ВНИЗ по картинке,
    поэтому y_tag направляем ПРОТИВ face_up.
    """
    z = norm(face_normal)
    y = -norm(face_up)
    x = np.cross(y, z)
    x = norm(x)
    y = np.cross(z, x)

    if abs(yaw_deg) > 1e-9:
        Rz = rot_axis_angle(z, yaw_deg)
        x = Rz @ x
        y = Rz @ y

    return make_T_from_axes_and_pos(x, y, z, face_center)

