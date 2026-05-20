#!/bin/bash

# Math Explanation API Server Startup Script

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set"
    echo "Please copy .env.example to .env and fill in your API keys"
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