# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with trusted hosts to avoid hash errors
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application files
COPY . .

# Create model directories
RUN mkdir -p model quantized_model

# Expose the port Streamlit runs on
EXPOSE 8501

# Health check for container
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Entry point: Quantize model, then run Streamlit app
CMD ["sh", "-c", "python quantize.py && streamlit run app.py"]
