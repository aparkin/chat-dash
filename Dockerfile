FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for building packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ChatDash.py .
COPY assets/ ./assets/
COPY data/ ./data/

# Expose port 8051
EXPOSE 8051

# Command to run ChatDash.py
CMD ["python", "ChatDash.py"]
