# Nanoleaf Gesture Control

## 🧠 Introduction

**Nanoleaf Gesture Control** is a computer vision-based application that enables users to control the brightness of Nanoleaf lights using hand gestures and head orientation. Utilizing MediaPipe's hand and face tracking capabilities, this project offers an intuitive and touchless interface for smart lighting.

## 📑 Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Dependencies](#dependencies)
* [Configuration](#configuration)
* [Documentation](#documentation)
* [Examples](#examples)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

## ⚙️ Installation

1. Clone this repository and navigate to the directory:

   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the required MediaPipe models and place them in the appropriate locations:

   * `hand_landmarker.task`
   * `gesture_recognizer.task`
   * `models/face_landmarker.task`

## ▶️ Usage

Run the main application:

```bash
python main.py
```

You’ll see two video windows:

* **Live Gesture Control**: Visualizes pinch gestures for brightness adjustment.
* **Face Orientation**: Displays head orientation and its status for gesture validation.

## ✨ Features

* **Pinch to Adjust Brightness**: Adjust light brightness by pinching thumb and index finger while the pinky is raised.
* **Head Orientation Gating**: Changes are applied only if your head is oriented within a predefined range.
* **Gesture Recognition**: Determines the active hand (left or right) using MediaPipe.
* **Hot Reloading**: `gesture.py` updates automatically without restarting the app.
* **Nanoleaf Integration**: Real-time control over Nanoleaf brightness via HTTP API.

## 📦 Dependencies

* `opencv-python`
* `mediapipe`
* `requests`

Install via:

```bash
pip install opencv-python mediapipe requests
```



## 🔧 Configuration

Edit `nanoleaf_utils.py` to configure your Nanoleaf device:

```python
NANO_IP    = "192.168.2.34"
AUTH_TOKEN = "your_auth_token_here"
```

Thresholds and tuning values can be modified in:

* `gesture.py` for pinch sensitivity.
* `face_orientation.py` for head yaw/roll configuration.

## 📚 Documentation

Modules and their roles:

* `main.py`: Application entry point and event loop.
* `gesture.py`: Pinch detection and brightness computation.
* `gesture_recognizer.py`: Determines which hand is in use.
* `face_orientation.py`: Evaluates head pose using key facial landmarks.
* `nanoleaf_utils.py`: Contains helper functions to interact with the Nanoleaf API.
* `nanoleaf_gesture_control.py`: An earlier standalone implementation combining brightness control with gesture detection.
* `HandTrackingModule.py`: A MediaPipe demo for drawing hand landmarks.

## 🧪 Examples

* Raise your pinky and pinch your fingers to set brightness.
* Look straight ahead to enable updates.
* Turn your head outside the allowed yaw range to pause brightness changes.

## 🛠️ Troubleshooting

* **Model Not Found**: Ensure all `.task` model files are correctly placed.
* **Camera Not Detected**: Check that your webcam is properly connected and accessible.
* **Brightness Not Updating**: Make sure pinky is raised, and face is oriented correctly.

## 👥 Contributors

* **You** – Feel free to expand this section as collaborators join.

## 📄 License

This project is licensed under the MIT License.

Let me know if you'd like this exported as a `.md` file or if you want to include diagrams, screenshots, or installation scripts.
