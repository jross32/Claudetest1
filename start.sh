#!/usr/bin/env bash
set -e

# Activate venv if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Copy .env.example if no .env exists
if [ ! -f .env ] && [ -f .env.example ]; then
  echo "⚠️  No .env found. Copy .env.example and add your ANTHROPIC_API_KEY."
  exit 1
fi

echo "🚀 Starting AI Coder Dashboard on http://localhost:8000"
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
