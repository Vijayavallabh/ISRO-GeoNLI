#!/bin/bash

SESSION_NAME="isro-geo-api"
APP_CMD="uvicorn app_prod:app --host 0.0.0.0 --port 8080"
LOG_FILE="server.log"

echo "Starting ISRO-GeoNLI API in PRODUCTION mode..."
echo
echo "Features:"
echo "  - Models preloaded on startup"
echo "  - No auto-reload"
echo "  - Runs inside tmux session: $SESSION_NAME"
echo "  - Auto-restart on crash"
echo "  - Logs -> $LOG_FILE"
echo
echo "This will take 30–60 seconds to load models..."
echo

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "✅ tmux session '$SESSION_NAME' already running."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

# Start tmux session
tmux new-session -d -s "$SESSION_NAME"

# Send auto-restart loop to tmux
tmux send-keys -t "$SESSION_NAME" "
echo '==== ISRO-GeoNLI PROD SERVER STARTED ===='
while true; do
    echo \"[\$(date)] Starting server...\" | tee -a $LOG_FILE
    $APP_CMD >> $LOG_FILE 2>&1
    echo \"[\$(date)] Server stopped / crashed. Restarting in 5s...\" | tee -a $LOG_FILE
    sleep 5
done
" C-m

echo "✅ Server started in tmux session '$SESSION_NAME'"
echo "➡ Attach using: tmux attach -t $SESSION_NAME"
echo "➡ Detach using: Ctrl+B then D"
