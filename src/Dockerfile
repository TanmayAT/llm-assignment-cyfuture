FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY /src/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "src.endpoint:builder", "--host", "0.0.0.0", "--port", "8000"]