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
COPY . .

# Expose API port
EXPOSE 8000

# Set environment variables (example)
ENV PYTHONUNBUFFERED=1

# Entry point script
CMD ["sh", "-c", "python scripts/build_knowledge_base.py && uvicorn api.main:app --host 0.0.0.0 --port 8000"]
