FROM python:3.11-slim

# Install system dependencies
# git and build-essential might be needed for some python extensions
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY mind/requirements.txt .

# Install dependencies
# Upgrade pip to ensure we get the latest wheels
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY . .

# Set python path so 'mind' module is found
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "mind/run_backtest.py"]
