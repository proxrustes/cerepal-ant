from typing import Optional
import cv2
import numpy as np


def draw_text_panel(
    img: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 10),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.7,
    thickness: int = 2,
    line_gap: int = 6,
    padding: int = 10,
    bg_alpha: float = 0.55,
    bg_color: tuple[int, int, int] = (0, 0, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    max_width: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """
    Рисует полупрозрачную панель с текстом.
    Возвращает bbox панели: (x1, y1, x2, y2).
    """
    if not lines:
        return (0, 0, 0, 0)

    h, w = img.shape[:2]
    x0, y0 = origin

    if max_width is None:
        max_width = w - x0 - 10

    # 1) измеряем строки, при необходимости обрезаем по ширине
    measured: list[tuple[str, tuple[int, int]]] = []
    max_line_w = 0
    total_h = 0

    for s in lines:
        # мягкая обрезка по пикселям (просто режем строку, пока не влезет)
        ss = str(s)
        (tw, th), baseline = cv2.getTextSize(ss, font, font_scale, thickness)
        if tw > max_width:
            # обрежем и добавим "..."
            ell = "..."
            while ss and cv2.getTextSize(ss + ell, font, font_scale, thickness)[0][0] > max_width:
                ss = ss[:-1]
            ss = (ss + ell) if ss else ell
            (tw, th), baseline = cv2.getTextSize(ss, font, font_scale, thickness)

        measured.append((ss, (tw, th + baseline)))
        max_line_w = max(max_line_w, tw)
        total_h += (th + baseline) + line_gap

    total_h -= line_gap  # убираем последний gap

    # 2) bbox панели
    x1 = x0
    y1 = y0
    x2 = min(w - 1, x0 + max_line_w + 2 * padding)
    y2 = min(h - 1, y0 + total_h + 2 * padding)

    # 3) рисуем полупрозрачный фон
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, bg_alpha, img, 1.0 - bg_alpha, 0, img)

    # 4) печатаем строки
    y = y1 + padding
    for ss, (_tw, thb) in measured:
        thb_int = int(thb)
        y_text = y + thb_int - 2  # небольшой тюнинг baseline
        cv2.putText(img, ss, (x1 + padding, y_text), font, font_scale, text_color, thickness, cv2.LINE_AA)
        y += thb_int + line_gap

    return (x1, y1, x2, y2)
