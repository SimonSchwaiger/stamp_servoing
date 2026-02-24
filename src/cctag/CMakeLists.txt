cmake_minimum_required(VERSION 3.8)
project(cctag)

# ---------- ROS 2 / ament ----------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

# ---------- Deterministic CCTag discovery ----------
set(_CCTAG_PREFIX "${CMAKE_CURRENT_LIST_DIR}/third_party/install/CCTag")
set(CCTag_DIR "${_CCTAG_PREFIX}/lib/cmake/CCTag" CACHE PATH "Path to CCTagConfig.cmake" FORCE)

if(NOT EXISTS "${CCTag_DIR}/CCTagConfig.cmake")
  message(FATAL_ERROR
    "CCTag not found.\n"
    "Expected: ${CCTag_DIR}/CCTagConfig.cmake\n"
    "Run: src/cctag/scripts/install_cctag.sh before colcon build.\n"
    "Expected install prefix: ${_CCTAG_PREFIX}"
  )
endif()

find_package(CCTag CONFIG REQUIRED)

# ---------- Node ----------
add_executable(cctag_detector_node src/cctag_detector_node.cpp)

ament_target_dependencies(cctag_detector_node
  rclcpp
  sensor_msgs
  vision_msgs
  cv_bridge
)

# IMPORTANT: use plain signature to match ament_target_dependencies()
target_link_libraries(cctag_detector_node
  CCTag::CCTag
  ${OpenCV_LIBRARIES}
)

# Runtime lookup for non-system prefix
set_target_properties(cctag_detector_node PROPERTIES
  BUILD_RPATH   "${_CCTAG_PREFIX}/lib"
  INSTALL_RPATH "${_CCTAG_PREFIX}/lib"
)

install(TARGETS cctag_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()