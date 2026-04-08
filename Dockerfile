# Use high-performance slim-bullseye base for rapid deployment
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 80

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Ensure the app follows the standardized entrypoint
EXPOSE 80

# Launch the FastAPI server
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "80"]
