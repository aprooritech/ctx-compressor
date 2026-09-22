from __future__ import annotations

import uvicorn

from src.app import create_app
from src.config import get_settings

app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    run()
