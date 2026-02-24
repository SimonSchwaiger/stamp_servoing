## Install and Run OAK
apt-get update && apt-get install -y --no-install-recommends \
    cmake libpcl-dev libeigen3-dev \
    libvulkan1 libxcb-randr0 mesa-vulkan-drivers \
    ros-${ROS_DISTRO}-depthai-ros \
    ros-${ROS_DISTRO}-rviz2

ros2 launch depthai_ros_driver camera.launch.py

## Install and Run Marker Detection
cd ${ROS2_WS}
rosdep install --from-paths src --ignore-src -r -y
colcon build
. install/setup.bash
ros2 run cctag cctag_detector_node --ros-args \
  -p image_topic:=/oak/rgb/image_raw