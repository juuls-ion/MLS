#!/bin/bash

# Time to give port 8000 a break – we're about to clear it out!

# Find any process IDs using port 8000
PIDS=$(lsof -ti:8000)

if [ -z "$PIDS" ]; then
    echo "Port 8000 is already free. No mischief found here!"
else
    echo "Port 8000 is being hogged by process(es): $PIDS"
    echo "Killing them with extreme prejudice..."
    # Kill all processes using port 8000
    kill -9 $PIDS
    echo "Port 8000 is now as clear as a sunny day!"
fi
