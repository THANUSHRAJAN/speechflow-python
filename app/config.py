"""Runtime settings, read once from environment variables (and speech-service/.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SERVICE_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(SERVICE_ROOT / ".env")

# The browser sends mono 16-bit PCM at this rate. Whisper models expect 16 kHz input.
SAMPLE_RATE = 16_000

# Languages Whisper can transcribe (multilingual models). "auto" is handled separately.
SUPPORTED_LANGUAGES = frozenset(
    "en zh de es ru ko fr ja pt tr pl ca nl ar sv it id hi fi vi he uk el ms cs ro da hu ta no th "
    "ur hr bg lt la mi ml cy sk te fa lv bn sr az sl kn et mk br eu is hy ne mn bs kk sq sw gl mr "
    "pa si km sn yo so af oc ka be tg sd gu am yi lo uz fo ht ps tk nn mt sa lb my bo tl mg as tt "
    "haw ln ha ba jw su yue".split()
)

# Languages written without spaces between words: joined without a separator.
NO_SPACE_LANGUAGES = frozenset({"zh", "ja", "yue", "th", "lo", "km", "my"})


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    return int(_get_float(name, default))


@dataclass(frozen=True)
class Settings:
    model_name: str = "small"
    device: str = "auto"
    compute_type: str = "int8"
    model_dir: Path = SERVICE_ROOT / "models"
    beam_size: int = 1

    host: str = "127.0.0.1"
    port: int = 8000
    allowed_origins: tuple[str, ...] = ("http://localhost:5173", "http://127.0.0.1:5173")
    log_level: str = "INFO"

    # How often (in seconds of newly received audio) the buffer is re-transcribed.
    chunk_interval: float = 1.5
    # Text ending within this many seconds of the live edge may still change, so it stays "interim".
    stability_margin: float = 1.0
    # If no text has stabilised after this many seconds, force a commit so the buffer cannot grow forever.
    max_buffer: float = 15.0
    # Audio kept after a forced commit so a word cut in half is heard again in the next window.
    overlap: float = 1.0
    # Buffers quieter than this (RMS, 0..1) are treated as silence and never sent to the model.
    silence_rms: float = 0.004
    # Do not transcribe buffers shorter than this.
    min_audio: float = 0.8

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = Path(os.getenv("WHISPER_MODEL_DIR", "./models"))
        if not model_dir.is_absolute():
            model_dir = SERVICE_ROOT / model_dir
        origins = tuple(
            o.strip()
            for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if o.strip()
        )
        return cls(
            model_name=os.getenv("WHISPER_MODEL", "small").strip(),
            device=os.getenv("WHISPER_DEVICE", "auto").strip(),
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
            model_dir=model_dir,
            beam_size=max(1, _get_int("WHISPER_BEAM_SIZE", 1)),
            host=os.getenv("HOST", "127.0.0.1").strip(),
            port=_get_int("PORT", 8000),
            allowed_origins=origins,
            log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
            chunk_interval=_get_float("CHUNK_INTERVAL_SECONDS", 1.5),
            stability_margin=_get_float("STABILITY_MARGIN_SECONDS", 1.0),
            max_buffer=_get_float("MAX_BUFFER_SECONDS", 15.0),
            overlap=_get_float("OVERLAP_SECONDS", 1.0),
        )
