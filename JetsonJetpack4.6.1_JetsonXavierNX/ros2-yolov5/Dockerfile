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
    vim \
    nano \
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

# Install ROS2 Dashing
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    wget https://github.com/ros2/ros2/releases/download/release-dashing-20210610/ros2-dashing-20210610-linux-bionic-arm64.tar.bz2 -P ~/Downloads && \
    mkdir -p ~/ros2_dashing && \
    cd ~/ros2_dashing && \
    tar xf ~/Downloads/ros2-dashing-20210610-linux-bionic-arm64.tar.bz2 && \
    rm -rf /var/lib/apt/lists/*

# Inisialisasi ROS2
RUN apt update && \
    apt install -y python-rosdep && \
    rosdep init && \
    rosdep update

# Install Dependensi ROS2
RUN rosdep install --from-paths ~/ros2_dashing/ros2-linux/share --ignore-src --rosdistro dashing -y --skip-keys "console_bridge fastcdr fastrtps libopensplice67 libopensplice69 osrf_testing_tools_cpp poco_vendor rmw_connext_cpp rosidl_typesupport_connext_c rosidl_typesupport_connext_cpp rti-connext-dds-5.3.1 tinyxml_vendor tinyxml2_vendor urdfdom urdfdom_headers"

# Install python3 library
RUN apt install -y libpython3-dev python3-pip && \
    pip3 install -U argcomplete

# Environment Setup 
CMD [".", "~/ros2_dashing/ros2-linux/setup.bash"]

# Install YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git && \
    cd yolov5 && \
    pip3 install -r requirements.txt

# Set environment variables for YOLOv5
ENV PYTHONPATH "${PYTHONPATH}:/yolov5"

# Entry point
CMD ["/bin/bash"]
