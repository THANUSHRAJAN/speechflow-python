import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SAMPLE_RATE, Settings  # noqa: E402
from app.transcriber import Segment, TranscriptionResult  # noqa: E402


class FakeEngine:
    """Returns scripted results so streaming logic is tested without a real model."""

    description = {"engine": "fake", "model": "fake"}

    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []  # (seconds_of_audio, language, prompt)
        self.loaded = True

    def load(self):
        self.loaded = True

    def transcribe(self, audio, language, prompt):
        self.calls.append((len(audio) / SAMPLE_RATE, language, prompt))
        if self.results:
            return self.results.pop(0)
        return TranscriptionResult([], language, 1.0)


def result(*segments, language="en", prob=0.95):
    return TranscriptionResult([Segment(s, e, t) for s, e, t in segments], language, prob)


def tone(seconds: float, amplitude: float = 0.2) -> bytes:
    """Loud-enough audio (a sine wave) as 16 kHz int16 PCM bytes."""
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (np.sin(2 * np.pi * 220 * t) * amplitude * 32767).astype("<i2").tobytes()


@pytest.fixture
def settings():
    return Settings(chunk_interval=1.0, stability_margin=1.0, max_buffer=8.0, overlap=1.0)
