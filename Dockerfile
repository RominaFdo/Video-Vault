# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies (if needed for your app)
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Pre-copy only requirements.txt for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Run app
CMD ["python", "app.py"]