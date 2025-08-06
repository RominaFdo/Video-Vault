# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Pre-copy only requirements.txt for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

# Expose port 7860 (Gradio default)
EXPOSE 7860

# Run app with proper host and port for Cloud Run
CMD ["python", "app.py"]