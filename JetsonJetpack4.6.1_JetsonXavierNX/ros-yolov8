# Menggunakan base image Ubuntu 18.04
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04

# Set timezone ke UTC
ENV TZ=UTC

# Install dependensi yang diperlukan
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    python3-opencv \
    curl \
    wget \
    git \
    libopencv-dev \
    libblas-dev \
    libffi-dev \
    liblapack-dev \
    libssl-dev \
    libxml2-dev \
    libxslt-dev \
    zlib1g-dev \
    gnupg2 \
    lsb-release \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision torchaudio

# Install TensorFlow
RUN pip3 install tensorflow

# Install ROS Melodic Desktop
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-melodic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# Install PyCuda
RUN pip3 install pycuda

# Install numpy, pandas, numba, Cupy
RUN pip3 install numpy pandas numba cupy

# Install YOLOv8
RUN pip3 install ultralytics

# Entry point
CMD ["/bin/bash"]
