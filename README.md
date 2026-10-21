# SpeechFlow speech service

Local, offline speech-to-text for SpeechFlow, built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(Whisper running on CTranslate2). Audio never leaves your computer.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # macOS/Linux: cp .env.example .env
python -m app.download_model        # optional: pre-download the model
python run.py
```

Health check: <http://localhost:8000/health> returns `{"status":"ok", ...}` once the model is loaded.

## Interface

WebSocket `ws://localhost:8000/ws/transcribe` (full protocol in `app/main.py`):

| Direction | Message |
|-----------|---------|
| client → server | `{"type":"config","language":"auto"\|"en"\|"hi"…}` first, then binary 16 kHz mono int16 PCM, then `{"type":"stop"}` |
| server → client | `ready`, `partial` (interim, replaceable), `final` (committed), `done`, `error` |

To swap in another local model, implement the small `SpeechEngine` protocol in `app/transcriber.py`
(`load()` and `transcribe(audio, language, prompt)`) and pass it to `create_app`.

## Model size, memory and speed

Change `WHISPER_MODEL` in `.env` (restart the service). All sizes are multilingual.

| Model | Download | RAM (CPU, int8) | Notes |
|-------|----------|-----------------|-------|
| `tiny` | ~75 MB | ~1 GB | fastest; weak on Hindi and other non-English languages |
| `base` | ~145 MB | ~1 GB | use on slow machines |
| `small` (default) | ~480 MB | ~2 GB | good balance for English and Hindi |
| `medium` | ~1.5 GB | ~4 GB | noticeably better accuracy, slower on CPU |
| `large-v3` | ~3 GB | ~6 GB+ | best accuracy; use an NVIDIA GPU (`WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`) |

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```
The tests use a scripted fake engine, so they run without downloading a model.
