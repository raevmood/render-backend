# Use official Python image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose API port
EXPOSE 8000

ENV PYTHONPATH=/app

# Entry point script
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000"]
