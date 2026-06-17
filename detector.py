import os
import socket
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import config

_HEAD_TOP_INDICES = (10, 338, 297, 332, 284)
_DOWNLOAD_CHUNK = 64 * 1024
_DOWNLOAD_TIMEOUT = 15


def _ensure_model(progress=None) -> str:
    model_dir = Path(config.MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / config.MODEL_FILENAME
    part_path = model_dir / (config.MODEL_FILENAME + ".part")

    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        return str(model_path)

    if part_path.exists():
        try:
            part_path.unlink()
        except OSError:
            pass

    print(f"[detector] downloading model to {model_path} ...")

    downloaded = 0
    with open(part_path, "wb") as f:
        req = urllib.request.Request(
            config.MODEL_URL,
            headers={"User-Agent": "head-guess/1.0"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT)
        except (urllib.error.URLError, socket.timeout) as e:
            raise RuntimeError(f"model download failed: {e}") from e

        total = resp.headers.get("Content-Length")
        total_i = int(total) if total and total.isdigit() else 0
        last_pct = -1
        while True:
            chunk = resp.read(_DOWNLOAD_CHUNK)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if progress is not None and total_i > 0:
                pct = int(downloaded * 100 / total_i)
                if pct != last_pct:
                    last_pct = pct
                    progress(downloaded, total_i)

    if downloaded < 1_000_000:
        try:
            part_path.unlink()
        except OSError:
            pass
        raise RuntimeError("model download incomplete (truncated)")

    os.replace(part_path, model_path)
    print("[detector] model ready.")
    return str(model_path)


class HeadDetector:
    def __init__(self, progress=None) -> None:
        model_path = _ensure_model(progress)
        base_opts = mp.tasks.BaseOptions(model_asset_path=model_path)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self._ts_ms = 0

    def detect(self, bgr_frame: np.ndarray):
        self._ts_ms += 33
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._ts_ms)
        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        h, w = bgr_frame.shape[:2]
        xs, ys = [], []
        for i in _HEAD_TOP_INDICES:
            if i < len(lm):
                xs.append(lm[i].x * w)
                ys.append(lm[i].y * h)
        if not xs:
            return None
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass
