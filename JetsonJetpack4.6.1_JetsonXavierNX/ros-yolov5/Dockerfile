# Menggunakan base image Ubuntu 18.04
FROM nvidia/cuda:11.4.1-cudnn8-devel-ubuntu18.04

# Set timezone ke UTC
ENV TZ=UTC
RUN cd ~

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

# Instalasi Realsense SDK Python
RUN apt update && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE && \
    add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u && \
    apt-get install librealsense2-utils && \
    sudo apt-get install librealsense2-dev && \
    pip install pyrealsense2 && \
    cd ~ && \
    git clone https://github.com/IntelRealSense/librealsense.git && \
    cd librealsense && \
    mkdir build && \
    cd build && \
    cmake ../ -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python && \
    make -j4 && \
    make install && \
    export PYTHONPATH=$PYTHONPATH:/usr/local/lib 
    

# Instalasi Realsense D4xx SDK
RUN pip3 install pyrealsense2 && \
    pip3 install transformations && \
    pip3 install dronekit && \
    pip3 install apscheduler && \
    pip3 install pyserial && \
    pip3 install opencv-python && \
    apt install python3-opencv

# Install PyTorch
RUN pip3 install torch torchvision torchaudio

# Install TensorFlow
RUN pip3 install tensorflow

# Install PyCuda
RUN pip3 install pycuda

# Install numpy, pandas, numba, Cupy
RUN pip3 install numpy pandas numba cupy

# Install ROS Melodic Desktop
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y ros-melodic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# Initialize ROS
RUN rosdep init && \
    rosdep update

# Instalasi ROS Realsense
RUN apt-get install ros-melodic-realsense2-camera && \
    apt-get install ros-$ROS_DISTRO-realsense2-description && \
    mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws/src/ && \
    git clone https://github.com/IntelRealSense/realsense-ros.git && \
    cd realsense-ros/ && \
    git checkout `git tag | sort -V | grep -P "^2.\d+\.\d+" | tail -1` && \
    cd .. && \
    catkin_init_workspace && \
    cd .. && \
    catkin_make clean && \
    catkin_make -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release && \
    catkin_make install && \
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc && \
    source ~/.bashrc

# Realsense D4XX
RUN cd ~/catkin_ws/src && \
    git clone https://github.com/hoangthien94/vision_to_mavros.git && \
    cd vision_to_mavros/script && \
    chmod +x d4xx_to_mavlink.py && \
    chmod +x opencv_depth_filtering.py
    chmod +x t265_to_mavlink.py
    chmod +x t265_test_streams.py
    
# Install Dronekit Mavproxy
RUN pip install dronekit
    apt-get install python3-dev python3-opencv python3-wxgtk4.0 python3-pip python3-matplotlib python3-lxml python3-pygame
    pip3 install PyYAML mavproxy
    echo 'export PATH="$PATH:$HOME/.local/bin"' >> ~/.bashrc
    pip3 install mavproxy pymavlink --upgrade

# Install Mavros
RUN apt-get install ros-noetic-mavros ros-noetic-mavros-extras && \
    cd ~ && \
    wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh && \
    chmod a+x install_geographiclib_datasets.sh && \
    ./install_geographiclib_datasets.sh

# Install vision_to_mavros
RUN cd ~/catkin_ws/src && \
    cd .. && \
    catkin_make && \
    source ~/.bashrc

# Install YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git && \
    cd yolov5 && \
    pip3 install -r requirements.txt

# Set environment variables for YOLOv5
ENV PYTHONPATH "${PYTHONPATH}:/yolov5"

# Entry point
CMD ["chmod" "666" "/dev/ttyUSB0"]
CMD ["/bin/bash"]
