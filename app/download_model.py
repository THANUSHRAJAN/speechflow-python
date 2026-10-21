"""Download (and smoke-load) the configured Whisper model:  python -m app.download_model"""
import logging

from .config import Settings
from .transcriber import FasterWhisperEngine

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings.from_env()
    FasterWhisperEngine(settings).load()
    print(f"Model '{settings.model_name}' is ready in {settings.model_dir}")
