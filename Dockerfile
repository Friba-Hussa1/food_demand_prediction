FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads \
    /app/models/regression \
    /app/models/arima \
    /app/results/regression \
    /app/results/arima \
    /app/forecasts/final \
    /app/Autoencoder && \
    chmod -R 755 /app

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg
ENV PORT=8080

# Expose the port Cloud Run expects
EXPOSE 8080

# Health check for Cloud Run
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Run the application
CMD ["python", "app.py"]
