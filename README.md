# MASAC-Based Uncalibrated PBVS

This repository provides a **robosuite 1.5** based robotic reinforcement learning environment, integrated with **Ray 2.49.1** for distributed training and **ROS 2 Humble** support, enabling training and simulation for vision and force control tasks.

## Dependencies

The project requires the following environment and libraries (it is recommended to use `conda` or `venv` for a virtual environment):

- Python ≥ 3.10  
- [robosuite 1.5](https://robosuite.ai/)  
- [Ray 2.49.1](https://docs.ray.io/en/latest/)  
- [ROS 2 Humble](https://docs.ros.org/en/humble/)  
- numpy 1.26.4  
- torch 2.5.1  
- tensorboard 2.14.0  

## Usage
Training.

```bash
cd ray
python train_jointlimit.py
```

Via tf system to observe the training process.

```bash
cd sb3/ros2
colcon build
source install/setup.bash
ros2 run robosuite_tf robosuite_tf_node --ros-args -p channel:=chatbus_1
```

Evaluation / Validation.

```bash
cd ray
python eval.py
```

## Features

  -  Supports single-arm and multi-arm robotic environments

  -  Reinforcement learning tasks combining vision and force control

  -  Distributed training for improved efficiency

  -  ROS 2 interface for real robot integration