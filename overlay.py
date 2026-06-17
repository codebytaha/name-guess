import cv2
import numpy as np

import config


def _font(name: str):
    return {
        "HERSHEY_SIMPLEX": cv2.FONT_HERSHEY_SIMPLEX,
        "HERSHEY_PLAIN": cv2.FONT_HERSHEY_PLAIN,
        "HERSHEY_DUPLEX": cv2.FONT_HERSHEY_DUPLEX,
        "HERSHEY_COMPLEX": cv2.FONT_HERSHEY_COMPLEX,
        "HERSHEY_TRIPLEX": cv2.FONT_HERSHEY_TRIPLEX,
        "HERSHEY_SCRIPT_SIMPLEX": cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    }.get(name, cv2.FONT_HERSHEY_DUPLEX)


def draw_dimmed_background(frame: np.ndarray) -> np.ndarray:
    alpha = float(config.THEME["bg_dim_alpha"])
    ksize = int(config.THEME["bg_blur_ksize"])
    if ksize > 0 and ksize % 2 == 1:
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), 0)
    else:
        blurred = frame
    return cv2.addWeighted(blurred, alpha, frame, 1.0 - alpha, 0)


def _measure(text: str, font, scale: float, thickness: int):
    (w, h), _baseline = cv2.getTextSize(text, font, scale, thickness)
    return w, h


def draw_letter_with_glow(
    frame: np.ndarray,
    text: str,
    center_xy: tuple[int, int],
) -> None:
    theme = config.THEME
    letter = config.LETTER
    font = _font(letter["font"])
    scale = float(letter["scale"])
    thickness = int(letter["thickness"])

    tw, th = _measure(text, font, scale, thickness)
    cx, cy = center_xy
    x = int(cx - tw / 2)
    y = int(cy + th / 2)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.putText(mask, text, (x, y), font, scale, 255, thickness, cv2.LINE_AA)

    sigma = float(theme["glow_sigma"])
    if sigma > 0:
        glow_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        glow_bgr = np.zeros_like(frame, dtype=np.float32)
        glow_bgr[:] = theme["glow_color"]
        glow_layer = (glow_bgr * glow_mask[..., None].astype(np.float32) / 255.0) * 0.6
        frame[:] = np.clip(frame.astype(np.float32) + glow_layer, 0, 255).astype(np.uint8)

    out_thick = int(theme["outline_thickness"])
    if out_thick > 0:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in offsets:
            cv2.putText(
                frame, text, (x + dx * out_thick, y + dy * out_thick),
                font, scale, theme["outline_color"], out_thick, cv2.LINE_AA,
            )

    cv2.putText(frame, text, (x, y), font, scale, theme["letter_color"],
                thickness, cv2.LINE_AA)


def draw_reveal_flash(frame: np.ndarray, text: str, progress: float) -> None:
    theme = config.THEME
    letter = config.LETTER
    font = _font(letter["font"])

    flash_scale = max(8.0, letter["scale"] * 2.2)
    flash_thick = max(14, int(letter["thickness"] * 1.6))

    (tw, th) = _measure(text, font, flash_scale, flash_thick)
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    x = int(cx - tw / 2)
    y = int(cy + th / 2)

    alpha = 1.0 - progress
    overlay = frame.copy()
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.putText(mask, text, (x, y), font, flash_scale, 255, flash_thick, cv2.LINE_AA)
    sigma = max(20.0, float(theme["glow_sigma"]) * 2.0)
    glow_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
    glow_bgr = np.zeros_like(overlay, dtype=np.float32)
    glow_bgr[:] = theme["glow_color"]
    glow_layer = (glow_bgr * glow_mask[..., None].astype(np.float32) / 255.0) * (0.7 * alpha)
    overlay[:] = np.clip(overlay.astype(np.float32) + glow_layer, 0, 255).astype(np.uint8)
    cv2.putText(overlay, text, (x, y), font, flash_scale, theme["outline_color"],
                flash_thick + 6, cv2.LINE_AA)
    color = tuple(int(c * alpha + 255 * (1 - alpha)) for c in theme["letter_color"])
    cv2.putText(overlay, text, (x, y), font, flash_scale, color,
                flash_thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_hud(frame: np.ndarray, letter: str, locked: bool = False) -> None:
    h, w = frame.shape[:2]
    font = _font(config.LETTER["font"])
    if locked:
        hint = "[SPACE] new   [R] reveal   [U] unlock   [Q] quit"
    else:
        hint = "[SPACE] new   [L] lock   [R] reveal   [Q] quit"
    (tw, th), _ = cv2.getTextSize(hint, font, 0.6, 2)
    pad = 12
    cv2.rectangle(frame, (10, h - th - pad * 3), (10 + tw + pad * 2, h - 10),
                  (0, 0, 0), -1)
    cv2.putText(frame, hint, (10 + pad, h - pad * 2), font, 0.6,
                (220, 220, 220), 2, cv2.LINE_AA)


def draw_lock_indicator(frame: np.ndarray, head_xy, t_ms: int) -> None:
    """Pulsing green ring drawn under the locked target's head."""
    if head_xy is None:
        return
    cx, cy = int(head_xy[0]), int(head_xy[1])
    pulse = 0.5 + 0.5 * (1 + np.sin(t_ms * 0.006))
    radius = int(28 + pulse * 8)
    color_bgr = (60, 255, 120)
    cv2.circle(frame, (cx, cy), radius, color_bgr, 4, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), radius + 6, color_bgr, 2, cv2.LINE_AA)


def draw_status_badge(frame: np.ndarray, text: str, color_bgr) -> None:
    """Top-center badge, e.g. 'LOCKED' or 'TARGET LOST'."""
    h, w = frame.shape[:2]
    font = _font(config.LETTER["font"])
    scale, thick = 0.8, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    pad_x, pad_y = 18, 8
    x = (w - tw) // 2 - pad_x
    y = 20
    cv2.rectangle(frame, (x, y), (x + tw + pad_x * 2, y + th + pad_y * 2),
                  (0, 0, 0), -1)
    cv2.rectangle(frame, (x, y), (x + tw + pad_x * 2, y + th + pad_y * 2),
                  color_bgr, 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x + pad_x, y + th + pad_y - 2),
                font, scale, color_bgr, thick, cv2.LINE_AA)

