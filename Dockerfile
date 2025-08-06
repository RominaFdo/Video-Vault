# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app ./app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Let Cloud Run set PORT, but default to 7860 for local dev
ENV PORT=7860

# Expose the default port
EXPOSE 7860

# Run the Gradio app
CMD ["python", "app/app.py"]
