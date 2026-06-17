THEME = {
    "bg_dim_alpha": 0.55,
    "bg_blur_ksize": 35,
    "letter_color": (255, 220, 120),
    "glow_color": (120, 180, 255),
    "outline_color": (0, 0, 0),
    "outline_thickness": 6,
    "glow_sigma": 14,
}

LETTER = {
    "font": "HERSHEY_DUPLEX",
    "scale": 4.0,
    "thickness": 10,
    "offset_above_head_px": 110,
    "smoothing": 0.6,
}

CAMERA = {
    "index": 0,
    "width": 1280,
    "height": 720,
}

WINDOW = {
    "name": "HeadGuess",
    "exit_keys": (ord("q"), ord("Q"), 27),
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DIR = "models"
MODEL_FILENAME = "face_landmarker.task"
