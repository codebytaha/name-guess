import ctypes
import random
import time
import traceback

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2
import numpy as np

import config
from detector import HeadDetector
from overlay import (
    draw_dimmed_background,
    draw_hud,
    draw_letter_with_glow,
    draw_reveal_flash,
)


def _smooth(prev, new, factor: float):
    if prev is None or new is None:
        return new
    return (
        prev[0] * factor + new[0] * (1 - factor),
        prev[1] * factor + new[1] * (1 - factor),
    )


def _show_splash(text: str, win: str, w: int, h: int) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.2
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(
        img, text,
        ((w - tw) // 2, (h + th) // 2),
        font, scale, (200, 220, 255), thick, cv2.LINE_AA,
    )
    cv2.imshow(win, img)
    cv2.waitKey(1)


def _fatal(win: str, msg: str) -> None:
    img = np.zeros((480, 854, 3), dtype=np.uint8)
    lines = msg.splitlines()
    y = 40
    for line in lines:
        cv2.putText(img, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 255), 2, cv2.LINE_AA)
        y += 28
    cv2.imshow(win, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    win = config.WINDOW["name"]
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    _show_splash("Starting...", win, 854, 480)

    cap = cv2.VideoCapture(config.CAMERA["index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA["height"])
    if not cap.isOpened():
        _fatal(win, f"Could not open camera at index {config.CAMERA['index']}.")
        return 1

    ok, probe = cap.read()
    if not ok:
        _fatal(win, "Camera opened but returned no frames.")
        cap.release()
        return 1
    frame_h, frame_w = probe.shape[:2]

    def _model_progress(downloaded: int, total: int) -> None:
        pct = int(downloaded * 100 / total) if total else 0
        _show_splash(f"Downloading model... {pct}%", win, frame_w, frame_h)

    try:
        detector = HeadDetector(progress=_model_progress)
    except Exception as e:
        traceback.print_exc()
        _fatal(win, f"Failed to load face model:\n{e}\n\nCheck internet connection.")
        cap.release()
        return 1

    try:
        letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        display_pos = None
        reveal_until_ms = None
        reveal_letter = letter

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            try:
                detected = detector.detect(frame)
            except Exception as e:
                traceback.print_exc()
                _fatal(win, f"Detector error:\n{e}")
                break

            display_pos = _smooth(
                display_pos, detected, float(config.LETTER["smoothing"])
            )

            draw_dimmed_background(frame)
            if display_pos is not None:
                cx, cy = display_pos
                draw_letter_with_glow(
                    frame,
                    letter,
                    (int(cx), int(cy - config.LETTER["offset_above_head_px"])),
                )

            now_ms = int(time.monotonic() * 1000)
            if reveal_until_ms is not None:
                remaining = reveal_until_ms - now_ms
                if remaining <= 0:
                    reveal_until_ms = None
                else:
                    progress = 1.0 - (remaining / 1000.0)
                    draw_reveal_flash(frame, reveal_letter, min(1.0, max(0.0, progress)))

            draw_hud(frame, letter)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in config.WINDOW["exit_keys"]:
                break
            if key == ord(" "):
                letter = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                reveal_letter = letter
                reveal_until_ms = None
            elif key in (ord("r"), ord("R")):
                reveal_letter = letter
                reveal_until_ms = now_ms + 1000
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
