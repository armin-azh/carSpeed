# Car Speed

## Installation
### 1. Install conda
### 2. Create an environment
    
    conda create -n your_env_name python=3.7
### 3. Activate the environment
    conda acrivate your_env_name
### 4. Install Packages
#### 4.1 Install Torch
##### 4.1.1 GPU support
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
##### 4.1.2 CPU 
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
#### 4.2 Install Other necessary packages
    pip install -r project_root/requirements.txt

### 5. Command
#### 5.1 Video File
##### 5.1.1 Yolo + MonoDepth
    python project_root/manage.py --yolo_mono --in_file path/to/file.mp4 --ref_speed car_speed_value
##### 5.1.2 Yolo + PyDNet
    python project_root/manage.py --yolo_pyd_net --in_file path/to/file.mp4 --ref_speed car_speed_value

#### 5.2 Camera or WebCam
##### 5.2.1 Yolo + MonoDepth
    python project_root/manage.py --cam --ref_speed car_speed_value
##### 5.2.2 Yolo + PyDNet
    python project_root/manage.py --cam --in_file --ref_speed car_speed_value

## System Specificity

| Device      | Model |
| ----------- | ----------- |
| GPU       | Nvidia 1650 4G Geforce|
| CPU   | Core i5 - 4900f|
| RAM   | 16 G|
| OS   | Ubuntu 18.04 LTS|






    