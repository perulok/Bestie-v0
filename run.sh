#!/bin/bash
# ========================================
# Bestie MVP launcher (backend + frontend)
# Robust to being run from any directory
# ========================================

set -e

# Resolve repo root (directory where this script lives)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/bestie-backend"
FRONTEND_DIR="$SCRIPT_DIR/Bestie-chat-v0"
VENV_DIR="$BACKEND_DIR/venv"

# Preflight: free ports 5050 and 3000 if occupied
BACK_PORT=5050
FRONT_PORT=3000
if lsof -ti tcp:$BACK_PORT >/dev/null 2>&1; then
  echo "⚠️  Port $BACK_PORT busy; freeing it..."
  kill $(lsof -ti tcp:$BACK_PORT) 2>/dev/null || true
fi
if lsof -ti tcp:$FRONT_PORT >/dev/null 2>&1; then
  echo "⚠️  Port $FRONT_PORT busy; freeing it..."
  kill $(lsof -ti tcp:$FRONT_PORT) 2>/dev/null || true
fi

# 1) Start backend
echo "🚀 Starting Bestie backend on port 5050..."
if [ ! -d "$BACKEND_DIR" ]; then
  echo "❌ Backend directory not found: $BACKEND_DIR" >&2
  exit 1
fi

# Ensure venv exists
if [ ! -d "$VENV_DIR" ]; then
  echo "🔧 Creating Python venv for backend..."
  python3 -m venv "$VENV_DIR"
fi

# Activate venv and ensure deps
source "$VENV_DIR/bin/activate"
python3 -c "import fastapi,uvicorn,dotenv,openai" 2>/dev/null || {
  echo "📦 Installing backend dependencies..."
  pip install --upgrade pip >/dev/null
  pip install fastapi uvicorn python-dotenv openai >/dev/null
}

cd "$BACKEND_DIR"
python3 -m uvicorn backend:app --reload --host 0.0.0.0 --port $BACK_PORT &
BACK_PID=$!
cd "$SCRIPT_DIR"

# 2) Start frontend
echo "🌐 Starting Bestie frontend on port 3000..."
if [ ! -d "$FRONTEND_DIR" ]; then
  echo "❌ Frontend directory not found: $FRONTEND_DIR" >&2
  kill $BACK_PID 2>/dev/null || true
  exit 1
fi

cd "$FRONTEND_DIR"
python3 -m http.server $FRONT_PORT &
FRONT_PID=$!
cd "$SCRIPT_DIR"

# 3) Trap CTRL+C to kill both
trap "echo '🛑 Shutting down...'; kill $BACK_PID $FRONT_PID 2>/dev/null || true" INT

# 4) Wait (keep terminal open while servers run)
wait
