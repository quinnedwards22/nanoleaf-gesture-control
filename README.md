# Nanoleaf Gesture Control

Real-time hand gesture and head pose control for **Nanoleaf Shapes** lights using a webcam.

## How It Works

Three signals must all be active simultaneously before a brightness update is sent:

| Signal | Module | Condition |
|--------|--------|-----------|
| **Pinch distance** | `src/gesture.py` | Thumb-to-index distance → brightness 0–100%. Pinky must be raised to activate. |
| **Right hand** | `src/gesture_recognizer.py` | Only the right hand triggers control. |
| **Head pose** | `src/face_orientation.py` | Head must face roughly forward-right (yaw −2° to +7°, roll < 20°). |

When all three conditions are met, brightness is sent to the Nanoleaf HTTP API.

```
webcam frame
    ├── gesture.py            → pinch brightness + pinky gate
    ├── gesture_recognizer.py → hand side (left / right)
    └── face_orientation.py   → head pose gate
                                      ↓ all true
                                nanoleaf_utils.py → HTTP PUT /state
```

## Project Structure

```
nanoleaf-gesture-control/
├── src/
│   ├── config.py               # Env var loading + model paths
│   ├── gesture.py              # Pinch-to-brightness, pinky gate
│   ├── gesture_recognizer.py   # Left/right hand detection
│   ├── face_orientation.py     # Head pose estimation
│   └── nanoleaf_utils.py       # Nanoleaf HTTP API wrapper
├── models/
│   ├── hand_landmarker.task    # MediaPipe hand landmark model
│   ├── gesture_recognizer.task # MediaPipe gesture classifier
│   └── face_landmarker.task    # MediaPipe face landmark model
├── main.py                     # Entry point — orchestrates all modules
├── requirements.txt
├── .env.example                # Credential template
├── Dockerfile
└── docker-compose.yml
```

## Prerequisites

- Python 3.11+
- A webcam
- Nanoleaf Shapes on your LAN with the HTTP API enabled
- Your Nanoleaf device IP and API auth token

### Getting a Nanoleaf API token

Hold the power button on your Nanoleaf controller for 5–7 seconds until the lights flash, then run:

```bash
curl -X POST http://<NANO_IP>:16021/api/v1/new
```

The response contains your `auth_token`.

## Local Setup

```bash
git clone <repo-url>
cd nanoleaf-gesture-control

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in NANO_IP and AUTH_TOKEN
```

Run:

```bash
python main.py
```

Two windows will open: **Live Gesture Control** (hand landmarks) and **Face Orientation** (pose debug).

## Docker Setup

> **Note:** Docker on Linux is the recommended deployment target. On Windows (WSL2), webcam
> passthrough requires [usbipd-win](https://github.com/dorssel/usbipd-win) and display requires
> [WSLg](https://github.com/microsoft/wslg).

```bash
cp .env.example .env
# Edit .env with your credentials

docker compose up --build
```

On Linux, allow Docker to open X11 windows first:

```bash
xhost +local:docker
docker compose up --build
```

## Configuration

### Required — `.env` file

| Variable | Description |
|----------|-------------|
| `NANO_IP` | IP address of your Nanoleaf controller (e.g. `192.168.1.50`) |
| `AUTH_TOKEN` | Nanoleaf API auth token |

### Tunable constants

| File | Constant | Default | Description |
|------|----------|---------|-------------|
| `src/gesture.py` | `MAX_PINCH_DIST` | `200` | Pixel distance mapping to 100% brightness |
| `src/gesture.py` | `UPDATE_INTERVAL` | `0.2` | Seconds between brightness updates |
| `src/face_orientation.py` | `YAW_MIN` / `YAW_MAX` | `-2` / `7` | Head yaw range (degrees) for activation |
| `src/face_orientation.py` | `MAX_ABS_ROLL_DEG` | `20.0` | Max head tilt before disabling control |
| `src/face_orientation.py` | `MIN_FACE_PRESENCE` | `0.5` | Minimum face detection confidence |

## Troubleshooting

**Camera not opening**
- Check that no other app is using the webcam.
- On Linux/Docker: ensure `/dev/video0` exists and the container has `devices` access.

**Cannot reach Nanoleaf**
- Verify `NANO_IP` is correct and the device is on the same LAN.
- Confirm the API is enabled: `curl http://<NANO_IP>:16021/api/v1/<AUTH_TOKEN>/state`

**Model file not found**
- Ensure all three `.task` files are present in the `models/` directory.
- Download links:
  - [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
  - [gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)
  - [face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

**No display / `cv2.error` on Docker**
- Run `xhost +local:docker` on the host before starting the container.
- On Windows WSL2 without WSLg, OpenCV display is not supported — run locally instead.
