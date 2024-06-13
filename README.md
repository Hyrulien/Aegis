# **Aegis**

## damage analysis bot by Hyrulien

### Introduction

Aegis is a damage analysis bot designed to provide accurate and efficient damage calculations using various libraries such as OpenCV, NumPy, EasyOCR, and more.

### Installation

To get started with Aegis, follow these steps to install the necessary dependencies.

#### Step 1: Install CUDA

Download and install CUDA from the following link:
[CUDA 12.1.0 Download](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)

#### Step 2: Install PyTorch

Run the following command to install PyTorch with CUDA 12.1 support:

```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
##### Step 3: Download these libraries:

```sh
pip install opencv-python
pip install numpy
pip install easyocr
pip install pyfiglet
```

###### Step 5: Place a .mp4 video of your testings into the folder \TestingVideos\

####### Step 6: Run bot using:

```sh
py aegis.py
```

output will be in \results\ in a .txt leading with the name of the video inputted in \TestingVideos\
