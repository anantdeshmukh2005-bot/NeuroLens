# ==========================================
# NeuroLens — Production Streamlit Container
# Multi-class Brain Tumor MRI Diagnostic Classifier
# ==========================================

FROM python:3.12-slim-bookworm

# System configuration & environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8501 \
    MPLCONFIGDIR=/tmp/matplotlib

# Install runtime system dependencies for OpenCV and health check curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (leveraging layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy production runtime files
COPY app.py .
COPY assets/ ./assets/
COPY scripts/gradcam_explainer.py ./scripts/
COPY models/mobilenet_two_stage_final_unweighted.keras ./models/
COPY .streamlit/ ./.streamlit/
COPY README.md .

# Create unprivileged runtime user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /tmp/matplotlib && \
    chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

EXPOSE 8501

# Health check probing Streamlit's native health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Launch Streamlit with dynamic cloud/local port support
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
