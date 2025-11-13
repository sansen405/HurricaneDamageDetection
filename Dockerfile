# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install dependencies directly
RUN pip install --no-cache-dir flask>=2.0.0 pillow>=9.0.0 tensorflow>=2.10.0 numpy>=1.21.0

# Copy application code and artifacts
COPY inference_server.py .
COPY artifacts/ ./artifacts/

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "inference_server.py"]

