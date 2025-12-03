#!/bin/bash
# Development server - fast startup, auto-reload, localhost only

echo "Starting ISRO-GeoNLI API in DEVELOPMENT mode..."
echo
echo "Features:"
echo "  - Auto-reload on code changes"
echo "  - Lazy model loading (faster startup)"
echo "  - Detailed error messages"
echo "  - Running on http://127.0.0.1:8000"
echo

uvicorn app_dev:app --reload --host 127.0.0.1 --port 8000
