# Gunakan TensorFlow resmi dengan Python 3 (tanpa GPU)
FROM tensorflow/tensorflow:latest

# Set environment variable untuk menghindari error CUDA
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
ENV TF_ENABLE_ONEDNN_OPTS=0

# Update paket dan install dependency tambahan
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    unzip \
    apt-utils \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install semua library yang dibutuhkan
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    tensorflow-hub \
    tensorflow-datasets \
    tflite-runtime \
    pillow \
    matplotlib

# Set direktori kerja di dalam container
WORKDIR /app

# Copy semua file dari host ke container
COPY . /app

# Jalankan script Python tanpa menampilkan error CUDA
# CMD python3 test_model_with_video.py 2>/dev/null
ENTRYPOINT ["/app/entrypoint.sh"]


