import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_1000"
marker_id = 23
side_px = 600  # размер маркера в пикселях
border_bits = 1

aruco = cv2.aruco
dictionary = (aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT_NAME))
              if hasattr(aruco, "getPredefinedDictionary")
              else aruco.Dictionary_get(getattr(aruco, ARUCO_DICT_NAME)))

img = np.zeros((side_px, side_px), dtype=np.uint8)

if hasattr(aruco, "generateImageMarker"):
    aruco.generateImageMarker(dictionary, marker_id, side_px, img, border_bits)
else:
    img = aruco.drawMarker(dictionary, marker_id, side_px, img, border_bits)

cv2.imwrite(f"aruco_{ARUCO_DICT_NAME}_{marker_id}.png", img)
print("saved")
