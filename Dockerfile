FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for the packages
# Keep it minimal to reduce image size
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code and model
COPY *.py ./
COPY core/ ./core/
COPY models/ ./models/

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Set Python path to include the current directory
ENV PYTHONPATH=/app

# Run the processing script
CMD ["python", "process_pdfs.py"]