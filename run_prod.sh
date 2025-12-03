#!/bin/bash
# Production server - preload models, optimized for performance

echo "Starting ISRO-GeoNLI API in PRODUCTION mode..."
echo
echo "Features:"
echo "  - Models preloaded on startup"
echo "  - No auto-reload"
echo "  - Running on http://0.0.0.0:8080"
echo
echo "This will take 30–60 seconds to load models..."
echo

uvicorn app_prod:app --host 0.0.0.0 --port 8080