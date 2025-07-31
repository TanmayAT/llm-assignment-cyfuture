FROM python:3.9-slim

WORKDIR /app

# Install only essential system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with optimizations
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && rm -rf ~/.cache/pip/*

# Copy only the src directory
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.endpoint:builder", "--host", "0.0.0.0", "--port", "8000"]