"""Start the service:  python run.py   (host/port come from .env)"""
import uvicorn

from app.config import Settings

if __name__ == "__main__":
    settings = Settings.from_env()
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, log_level=settings.log_level.lower())
