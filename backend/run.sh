#!/bin/bash

# Math Explanation API Server Startup Script

cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
fi

if [ -n "$LANGSMITH_API_KEY" ] && [ -z "$LANGCHAIN_API_KEY" ]; then
    export LANGCHAIN_API_KEY="$LANGSMITH_API_KEY"
fi

if [ -n "$LANGSMITH_PROJECT" ] && [ -z "$LANGCHAIN_PROJECT" ]; then
    export LANGCHAIN_PROJECT="$LANGSMITH_PROJECT"
fi

if [ "$LANGSMITH_TRACING" = "true" ] && [ -z "$LANGCHAIN_TRACING_V2" ]; then
    export LANGCHAIN_TRACING_V2=true
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ] && [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: neither OPENAI_API_KEY nor GOOGLE_API_KEY is set"
    echo "Please copy .env.example to .env and fill in at least one API key"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt -q

# Run the server
echo "Starting Math Explanation API on http://0.0.0.0:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
