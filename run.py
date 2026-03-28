#!/usr/bin/env python3
"""
run.py — Development / manual startup entry point.

Production: use install.sh which sets up systemd service.
Development: python run.py
"""

import uvicorn
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=1,
        log_level="info",
        reload=False,
    )
