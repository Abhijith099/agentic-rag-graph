# -------------------------------------------------------------
# Stage 1: Base Image & Environment Settings
# -------------------------------------------------------------
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system-level build tools needed for C-extensions (like asyncpg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------
# Stage 2: Install Python Dependencies (Layer Caching)
# -------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -------------------------------------------------------------
# Stage 3: Copy Application Code & Setup Non-Root Security User
# -------------------------------------------------------------
COPY . .

# Create a restricted system user with UID 10001
RUN useradd -m -u 10001 appuser && \
    chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose FastAPI port
EXPOSE 8058

# -------------------------------------------------------------
# Stage 4: Entrypoint
# -------------------------------------------------------------
CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8058"]
