xhost +local:docker

DOCKER_ARGS=()
DOCKER_ARGS+=("-e DISPLAY=$DISPLAY")
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")
DOCKER_ARGS+=("--device=/dev/dri:/dev/dri")
DOCKER_ARGS+=("-v $PWD/src:/opt/ros2_ws/src")
DOCKER_ARGS+=("-v $PWD/lab:/root/.jupyter/lab")
DOCKER_ARGS+=("-v $PWD/cache:/root/.cache") # Pytorch Cache

DOCKER_ARGS+=("--privileged -v /dev/bus/usb:/dev/bus/usb")
DOCKER_ARGS+=("--net host -v /dev:/dev --ipc=host -v /dev/shm:/dev/shm")

docker run -it --rm --name fhtw_ros ${DOCKER_ARGS[@]} \
    ghcr.io/simonschwaiger/ros-ml-container:ros2_humble_opensource bash
