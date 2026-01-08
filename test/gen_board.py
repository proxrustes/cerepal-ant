import cv2
import numpy as np

ARUCO_DICT_NAME = "DICT_4X4_1000"
BOARD_COLS = 6
BOARD_ROWS = 6
MARKER_LENGTH_M = 0.04      # 4 см
MARKER_SEPARATION_M = 0.01  # 1 см

DPI = 300  # печать
MARGIN_M = 0.01  # поля вокруг доски, 1 см

def m_to_px(m: float, dpi: int) -> int:
    # 1 inch = 0.0254 m
    return int(round(m * dpi / 0.0254))

aruco = cv2.aruco
dictionary = (aruco.getPredefinedDictionary(getattr(aruco, ARUCO_DICT_NAME))
              if hasattr(aruco, "getPredefinedDictionary")
              else aruco.Dictionary_get(getattr(aruco, ARUCO_DICT_NAME)))

# board object
if hasattr(aruco, "GridBoard"):
    board = aruco.GridBoard((BOARD_COLS, BOARD_ROWS), MARKER_LENGTH_M, MARKER_SEPARATION_M, dictionary)
else:
    board = aruco.GridBoard_create(BOARD_COLS, BOARD_ROWS, MARKER_LENGTH_M, MARKER_SEPARATION_M, dictionary)

marker_px = m_to_px(MARKER_LENGTH_M, DPI)
sep_px = m_to_px(MARKER_SEPARATION_M, DPI)
margin_px = m_to_px(MARGIN_M, DPI)

width_px  = BOARD_COLS * marker_px + (BOARD_COLS - 1) * sep_px + 2 * margin_px
height_px = BOARD_ROWS * marker_px + (BOARD_ROWS - 1) * sep_px + 2 * margin_px
out_size = (width_px, height_px)

# render
if hasattr(board, "generateImage"):
    img = board.generateImage(out_size, marginSize=margin_px, borderBits=1)
else:
    img = np.zeros((height_px, width_px), dtype=np.uint8)
    # drawPlanarBoard(board, outSize, img, marginSize, borderBits)
    aruco.drawPlanarBoard(board, out_size, img, margin_px, 1)

cv2.imwrite(f"gridboard_{BOARD_COLS}x{BOARD_ROWS}_{DPI}dpi.png", img)
print("saved:", out_size)
