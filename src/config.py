import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Absolute path to the models/ directory at the project root
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

# Nanoleaf credentials — set these in your .env file or environment
NANO_IP = os.environ.get("NANO_IP", "")
AUTH_TOKEN = os.environ.get("AUTH_TOKEN", "")
