# HeadGuess

A fullscreen, aesthetic letter-on-head guessing game in Python. A random capital letter floats above the detected head with a neon-style glow; the person behind you has to guess it. Cushion consequences are handled offline by your cousins.

Inspired by the paid Snapchat lens — built with MediaPipe Face Landmarker + OpenCV so you can play it locally for free.

## Demo loop

1. App opens fullscreen with a dimmed, blurred view of your webcam.
2. Sit in frame → a glowing capital letter appears above your forehead and follows your head smoothly.
3. The person behind you guesses the letter.
4. Press `Space` for a new random letter, `R` to reveal the answer, `Q` (or `Esc`) to quit.

## Controls

| Key     | Action               |
|---------|----------------------|
| `Space` | New random letter    |
| `R`     | Reveal (1s flash)    |
| `Q` / `Esc` | Quit             |

## Re-skinning (no code edits)

Edit `config.py` to change the look:

- `THEME["letter_color"]` — letter fill, BGR `(B, G, R)`
- `THEME["glow_color"]` — halo color, BGR
- `THEME["bg_dim_alpha"]` — 0.0 (no dim) to 1.0 (pitch black)
- `THEME["bg_blur_ksize"]` — odd int, 0 to disable background blur
- `LETTER["scale"]` / `LETTER["thickness"]` — letter size and weight
- `LETTER["smoothing"]` — 0.0 (snappy) to ~0.9 (very smooth, laggier)

Example pastel theme:
```python
THEME["letter_color"] = (255, 200, 230)   # soft pink
THEME["glow_color"]   = (200, 230, 255)   # light cyan halo
THEME["bg_dim_alpha"] = 0.4
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

On first launch the app downloads `face_landmarker.task` (~5 MB) to `models/` and shows a "Downloading model... N%" splash. The model is cached after the first run, so subsequent launches start instantly.

## Files

- `main.py` — game loop, fullscreen window, input
- `detector.py` — MediaPipe FaceLandmarker wrapper, lazy model download
- `overlay.py` — dim/blur background, glow text rendering, reveal flash
- `config.py` — theme / letter / camera / window knobs
- `requirements.txt` — pinned dependencies

## Notes

- Uses MediaPipe's **Tasks API** (`FaceLandmarker`) with `RunningMode.VIDEO`.
- Head-top point is landmark 10, averaged with adjacent top-ring points (338, 297, 332, 284) for stability. (Landmark 152 is the chin — common mistake.)
- Frame is mirrored *before* detection so landmark coords match the displayed selfie view directly.
- Tested with `mediapipe==0.10.35` and `opencv-python==4.13`.

## Out of scope (for now)

Scoring, sound effects, multiplayer, recording. The cleanest extension points are `main.py` (game state) and `overlay.draw_hud` (top-bar score banner).
