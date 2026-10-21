"""Local speech recognition: model wrapper plus the incremental (chunked) streaming logic.

Design
------
Whisper is not a streaming model: it transcribes a finished piece of audio. To get
near-real-time output we keep a rolling buffer of audio that has not been "committed" yet and
re-transcribe it every ``chunk_interval`` seconds:

* Segments that ended comfortably before the live edge (``stability_margin``) are unlikely to
  change, so they are *committed* (final text) and their audio is dropped from the buffer.
* Everything after that is *interim* text and is simply replaced on the next pass.
* If speech runs on without a pause for ``max_buffer`` seconds, a forced commit keeps the
  last ``overlap`` seconds of audio; the duplicate words this can create are removed by
  :func:`dedupe_overlap`.

The engine (model) is behind a tiny interface so another local model can replace Whisper.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import NO_SPACE_LANGUAGES, SAMPLE_RATE, Settings

logger = logging.getLogger("speechflow.transcriber")


# --------------------------------------------------------------------------- data types
@dataclass(frozen=True)
class Segment:
    start: float  # seconds from the start of the audio that was transcribed
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    segments: list[Segment]
    language: str | None
    language_probability: float


class SpeechEngine(Protocol):
    """Anything that can turn float32 mono 16 kHz audio into timestamped text."""

    @property
    def loaded(self) -> bool: ...

    @property
    def description(self) -> dict: ...

    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None) -> TranscriptionResult: ...


# --------------------------------------------------------------------------- Whisper engine
class FasterWhisperEngine:
    """faster-whisper (CTranslate2) model that is loaded once and reused for every chunk."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        self._lock = threading.Lock()  # one transcription at a time on the shared model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def description(self) -> dict:
        return {
            "engine": "faster-whisper",
            "model": self._settings.model_name,
            "device": self._settings.device,
            "computeType": self._settings.compute_type,
        }

    def load(self) -> None:
        # Imported here so unit tests and tooling can import this module without the heavy dependency.
        from faster_whisper import WhisperModel

        s = self._settings
        s.model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Loading Whisper model '%s' (device=%s, compute_type=%s). "
                    "The first run downloads the weights into %s",
                    s.model_name, s.device, s.compute_type, s.model_dir)
        started = time.perf_counter()
        self._model = WhisperModel(
            s.model_name,
            device=s.device,
            compute_type=s.compute_type,
            download_root=str(s.model_dir),
        )
        logger.info("Model ready in %.1fs", time.perf_counter() - started)

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str | None) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("Speech model is not loaded")
        with self._lock:
            segments, info = self._model.transcribe(
                audio,
                language=language,
                task="transcribe",
                beam_size=self._settings.beam_size,
                vad_filter=True,  # skips silence, which also prevents Whisper "hallucinating" text
                vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200},
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
            )
            result = [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments]
        return TranscriptionResult(result, info.language, float(info.language_probability))


# --------------------------------------------------------------------------- text helpers
def _normalize_word(word: str) -> str:
    """Lower-case and keep only letters, combining marks and digits (works for Devanagari etc.)."""
    return "".join(ch for ch in word.casefold() if unicodedata.category(ch)[0] in "LMN")


def dedupe_overlap(existing: str, new: str, min_words: int = 3, max_words: int = 12,
                   char_level: bool = False) -> str:
    """Remove from the start of ``new`` any words that repeat the end of ``existing``.

    Adjacent windows can overlap, so the same few words may be recognised twice. We look for the
    longest run (>= ``min_words``) where the tail of ``existing`` equals the head of ``new``,
    ignoring case and punctuation, and drop that run from ``new``.
    """
    new = new.strip()
    existing = existing.strip()
    if not existing or not new:
        return new

    ex_words, new_words = existing.split(), new.split()
    limit = min(max_words, len(ex_words), len(new_words))
    ex_norm = [_normalize_word(w) for w in ex_words[-limit:]] if limit else []
    new_norm = [_normalize_word(w) for w in new_words[:limit]] if limit else []
    for k in range(limit, max(min_words, 1) - 1, -1):
        if ex_norm[-k:] == new_norm[:k] and all(ex_norm[-k:]):
            return " ".join(new_words[k:])

    if char_level:  # languages without spaces: compare raw characters
        for k in range(min(30, len(existing), len(new)), 3, -1):
            if existing[-k:] == new[:k]:
                return new[k:]
    return new


def join_text(existing: str, addition: str, language: str | None) -> str:
    """Append ``addition`` to ``existing`` with the right separator for the language."""
    existing, addition = existing.strip(), addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    separator = "" if language in NO_SPACE_LANGUAGES else " "
    return existing + separator + addition


# --------------------------------------------------------------------------- streaming session
class StreamingSession:
    """State for one recording: buffers PCM audio and turns it into interim and final text."""

    def __init__(self, engine: SpeechEngine, settings: Settings, language: str = "auto") -> None:
        self._engine = engine
        self._s = settings
        self._locked = language != "auto"
        self._language: str | None = None if language == "auto" else language  # used for decoding
        self._detected: str | None = None
        self._buffer = bytearray()  # int16 little-endian PCM not yet committed
        self._unprocessed = 0  # samples received since the last update started
        self._committed = ""
        self._last_partial = ""

    # -- inspection ---------------------------------------------------------
    @property
    def full_text(self) -> str:
        return self._committed

    @property
    def language(self) -> str:
        return self._language or self._detected or "auto"

    @property
    def buffer_seconds(self) -> float:
        return len(self._buffer) / 2 / SAMPLE_RATE

    def ready_for_update(self) -> bool:
        return (self._unprocessed / SAMPLE_RATE >= self._s.chunk_interval
                and self.buffer_seconds >= self._s.min_audio)

    # -- input --------------------------------------------------------------
    def add_audio(self, data: bytes) -> None:
        if len(data) % 2:
            data = data[:-1]  # drop a stray byte; samples are 2 bytes each
        self._buffer += data
        self._unprocessed += len(data) // 2

    # -- processing ---------------------------------------------------------
    async def update(self, final: bool = False) -> list[dict]:
        """Transcribe the buffer and return the events (final / partial) to send to the client.

        The slow model call runs in a worker thread. Buffer and text state are only touched on
        the event loop, so audio that arrives during a transcription cannot corrupt them.
        """
        snapshot = bytes(self._buffer)
        n_samples = len(snapshot) // 2
        self._unprocessed = 0
        if n_samples == 0:
            return []

        audio = np.frombuffer(snapshot, dtype="<i2").astype(np.float32) / 32768.0
        if float(np.sqrt(np.mean(audio * audio))) < self._s.silence_rms:
            return self._handle_silence(n_samples, final)

        prompt = self._committed[-200:] or None  # recent text helps spelling and punctuation continuity
        started = time.perf_counter()
        result = await asyncio.to_thread(self._engine.transcribe, audio, self._language, prompt)
        logger.debug("Transcribed %.1fs of audio in %.2fs", n_samples / SAMPLE_RATE,
                     time.perf_counter() - started)
        return self._apply(result, n_samples, final)

    def _handle_silence(self, n_samples: int, final: bool) -> list[dict]:
        if final:
            self._buffer.clear()
            return []
        keep = int(SAMPLE_RATE * 1.0)  # keep the last second so a word onset is not cut off
        if n_samples > 3 * SAMPLE_RATE and len(self._buffer) // 2 > keep:
            del self._buffer[: (len(self._buffer) // 2 - keep) * 2]
        return []

    def _apply(self, result: TranscriptionResult, n_samples: int, final: bool) -> list[dict]:
        buf_dur = n_samples / SAMPLE_RATE
        segments = [s for s in result.segments if s.text.strip()]
        self._update_language(result, buf_dur, bool(segments))

        forced = False
        cut = 0
        if final:
            stable = segments
            cut = n_samples
        else:
            limit = buf_dur - self._s.stability_margin
            stable = []
            for seg in segments:  # segments are ordered, so the stable ones form a prefix
                if seg.end <= limit:
                    stable.append(seg)
                else:
                    break
            if stable:
                cut = int(stable[-1].end * SAMPLE_RATE)
            elif segments and buf_dur >= self._s.max_buffer:
                forced = True
                if len(segments) > 1:
                    stable = segments[:-1]
                    cut = int(stable[-1].end * SAMPLE_RATE)
                else:
                    stable = segments
                    cut = max(0, n_samples - int(self._s.overlap * SAMPLE_RATE))
        remaining = segments[len(stable):]

        events: list[dict] = []
        committed_text = self._join_segments([s.text for s in stable])
        if committed_text:
            added = self._commit(committed_text, forced)
            if added:
                events.append({"type": "final", "text": added, "fullText": self._committed,
                               "language": self.language})
        if cut:
            del self._buffer[: min(cut, len(self._buffer) // 2) * 2]

        if not final:
            partial = self._join_segments([s.text for s in remaining])
            self._last_partial = partial
            events.append({"type": "partial", "text": partial, "language": self.language})
        return events

    def _commit(self, text: str, forced: bool) -> str:
        """Append text to the committed transcript, removing words repeated from an overlap."""
        lang = self._language or self._detected
        text = dedupe_overlap(self._committed, text, min_words=1 if forced else 3,
                              char_level=lang in NO_SPACE_LANGUAGES)
        if not text:
            return ""
        self._committed = join_text(self._committed, text, lang)
        return text

    def _join_segments(self, texts: list[str]) -> str:
        lang = self._language or self._detected
        out = ""
        for text in texts:
            out = join_text(out, text, lang)
        return out

    def _update_language(self, result: TranscriptionResult, buf_dur: float, has_text: bool) -> None:
        if self._locked or not result.language:
            return
        self._detected = result.language
        # Lock the language once Whisper is confident, so later chunks do not flip languages.
        if has_text and buf_dur >= 2.0 and result.language_probability >= 0.6:
            self._language = result.language
            self._locked = True
            logger.info("Language locked to '%s' (p=%.2f)", result.language, result.language_probability)
