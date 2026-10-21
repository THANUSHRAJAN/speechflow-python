"""SpeechFlow speech service: a FastAPI app exposing a WebSocket for live transcription.

WebSocket protocol  (ws://HOST:PORT/ws/transcribe)
---------------------------------------------------
client -> server
    text   {"type": "config", "language": "auto" | "en" | "hi" | ...}     (first message)
    binary 16 kHz, mono, signed 16-bit little-endian PCM (any chunk size)
    text   {"type": "stop"}                                              (flush and finish)
server -> client
    {"type": "ready", "model": "...", "language": "..."}
    {"type": "partial", "text": "...", "language": "..."}                interim text, replaces the previous one
    {"type": "final", "text": "new text", "fullText": "all committed text", "language": "..."}
    {"type": "done", "fullText": "...", "language": "..."}               after "stop"
    {"type": "error", "message": "...", "fatal": bool}
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import SUPPORTED_LANGUAGES, Settings
from .transcriber import FasterWhisperEngine, SpeechEngine, StreamingSession

logger = logging.getLogger("speechflow.service")


def create_app(engine: SpeechEngine | None = None, settings: Settings | None = None) -> FastAPI:
    """Application factory. Tests pass in a fake engine; production loads faster-whisper."""
    settings = settings or Settings.from_env()
    logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    engine_given = engine is not None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or FasterWhisperEngine(settings)
        app.state.load_error = None
        if not app.state.engine.loaded and not engine_given:
            try:
                # Runs in a thread; the service starts accepting requests only once the model is ready.
                await asyncio.to_thread(app.state.engine.load)
            except Exception as exc:  # noqa: BLE001 - report any load failure through /health
                logger.exception("Could not load the speech model")
                app.state.load_error = str(exc)
        yield

    app = FastAPI(title="SpeechFlow speech service", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.allowed_origins),
                       allow_methods=["GET"], allow_headers=["*"])

    @app.get("/health")
    async def health():
        eng: SpeechEngine = app.state.engine
        if eng.loaded:
            return {"status": "ok", **eng.description}
        return JSONResponse(status_code=503, content={
            "status": "error",
            "detail": app.state.load_error or "Speech model is not loaded yet",
        })

    @app.websocket("/ws/transcribe")
    async def transcribe(ws: WebSocket):
        await ws.accept()
        eng: SpeechEngine = app.state.engine
        send_lock = asyncio.Lock()
        worker: asyncio.Task | None = None

        async def send(payload: dict) -> None:
            async with send_lock:
                await ws.send_json(payload)

        async def fail(message: str, code: int) -> None:
            try:
                await send({"type": "error", "message": message, "fatal": True})
                await ws.close(code=code)
            except Exception:  # noqa: BLE001 - client may already be gone
                pass

        if not eng.loaded:
            await fail("The speech model is not loaded. Check the speech service logs.", 1011)
            return

        try:
            try:
                first = await asyncio.wait_for(ws.receive_text(), timeout=10)
                config = json.loads(first)
            except (asyncio.TimeoutError, json.JSONDecodeError):
                await fail("Expected a JSON config message first.", 1008)
                return
            if not isinstance(config, dict) or config.get("type") != "config":
                await fail("Expected a JSON config message first.", 1008)
                return

            language = str(config.get("language") or "auto").lower()
            if language != "auto" and language not in SUPPORTED_LANGUAGES:
                await fail(f"Unsupported language code '{language}'.", 1008)
                return

            session = StreamingSession(eng, settings, language)
            await send({"type": "ready", "model": settings.model_name, "language": language})

            async def run_update() -> None:
                try:
                    for event in await session.update(final=False):
                        await send(event)
                except (WebSocketDisconnect, RuntimeError):
                    pass  # client went away mid-update
                except Exception:  # noqa: BLE001
                    logger.exception("Transcription step failed")
                    try:
                        await send({"type": "error", "fatal": False,
                                    "message": "A transcription step failed. Text so far is kept; "
                                               "see the speech service log for details."})
                    except Exception:  # noqa: BLE001
                        pass

            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break

                if message.get("bytes") is not None:
                    session.add_audio(message["bytes"])
                    if (worker is None or worker.done()) and session.ready_for_update():
                        worker = asyncio.create_task(run_update())
                    continue

                text = message.get("text")
                if text is None:
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    await send({"type": "error", "fatal": False, "message": "Ignored a malformed message."})
                    continue

                if isinstance(data, dict) and data.get("type") == "stop":
                    if worker is not None:
                        await worker  # let an in-flight update finish so state stays consistent
                    try:
                        for event in await session.update(final=True):
                            await send(event)
                    except Exception:  # noqa: BLE001
                        logger.exception("Final transcription step failed")
                        await send({"type": "error", "fatal": False,
                                    "message": "The last audio could not be transcribed."})
                    await send({"type": "done", "fullText": session.full_text, "language": session.language})
                    break
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception:  # noqa: BLE001
            logger.exception("WebSocket session crashed")
            await fail("The speech service hit an unexpected error.", 1011)
        finally:
            if worker is not None and not worker.done():
                worker.cancel()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001 - already closed
                pass

    return app


app = create_app()
