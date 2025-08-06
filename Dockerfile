# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Pre-copy only requirements.txt for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY app ./app

# Create a non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port (Cloud Run will set PORT env var)
EXPOSE 7860

# Run app
CMD ["python", "app/app.py"]