# Nanoleaf Gesture Control

Hand-gesture control for Nanoleaf Shapes using:

- **OpenCV** (camera capture)
- **MediaPipe** (hand‐landmark detection)
- **Requests** (Nanoleaf HTTP API)

## Features

- Swipe up/down → adjust brightness
- Swipe left/right → shift hue (0–360°)
- Rate-limited updates (0.5 s) to avoid API flooding

## Prerequisites

- Python 3.8+
- Nanoleaf Shapes on your LAN
- Nanoleaf API token & device IP

## Setup

1. Clone the repo  
   ```bash
   git clone git@github.com:YOUR_USERNAME/nanoleaf-gesture-control.git
   cd nanoleaf-gesture-control
