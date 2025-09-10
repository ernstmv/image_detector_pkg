#!/bin/bash

cd $HOME/ros2_ws

echo "[1/3] Building package\n"
colcon build --packages-select image_detector_pkg 

if [  $? -eq 1 ]; then
	echo "ERROR: Error building package. Please check"
	exit
else
	echo "Build done\n"
fi

echo
echo "[2/3] Sourcing ROS2 overlay\n"
source install/setup.bash

if [  $? -eq 1 ]; then
	echo "ERROR: Error sourcing environment. Please check"
	exit
else
	echo "Environment loaded\n"
fi

echo "[3/3] Launching package\n"
ros2 launch image_detector_pkg image_detector.launch.py
