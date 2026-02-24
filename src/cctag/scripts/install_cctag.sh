#!/usr/bin/env bash
set -euo pipefail

# Installs alicevision/CCTag into this ROS 2 package:
#   <pkg>/third_party/install/CCTag
# so that CMakeLists.txt can deterministically use:
#   CCTag_DIR=<pkg>/third_party/install/CCTag/lib/cmake/CCTag

PKG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

REPO_URL="https://github.com/alicevision/CCTag.git"
GIT_REF="${GIT_REF:-develop}"

SRC_DIR="${PKG_DIR}/third_party/CCTag"
BUILD_DIR="${PKG_DIR}/third_party/build/CCTag"
INSTALL_PREFIX="${PKG_DIR}/third_party/install/CCTag"

BUILD_TYPE="${BUILD_TYPE:-Release}"
CCTAG_WITH_CUDA="${CCTAG_WITH_CUDA:-OFF}"
CCTAG_BUILD_APPS="${CCTAG_BUILD_APPS:-OFF}"
CCTAG_BUILD_TESTS="${CCTAG_BUILD_TESTS:-OFF}"

mkdir -p "${PKG_DIR}/third_party" "${PKG_DIR}/third_party/build" "${PKG_DIR}/third_party/install"

if [[ ! -d "${SRC_DIR}/.git" ]]; then
  git clone "${REPO_URL}" "${SRC_DIR}"
fi

(
  cd "${SRC_DIR}"
  git fetch --all --tags
  git checkout "${GIT_REF}"
)

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DCCTAG_WITH_CUDA:BOOL="${CCTAG_WITH_CUDA}" \
  -DCCTAG_BUILD_APPS:BOOL="${CCTAG_BUILD_APPS}" \
  -DCCTAG_BUILD_TESTS:BOOL="${CCTAG_BUILD_TESTS}"

cmake --build "${BUILD_DIR}" -- -j"$(nproc)"
cmake --install "${BUILD_DIR}"

CCTAG_CMAKE_DIR="${INSTALL_PREFIX}/lib/cmake/CCTag"
if [[ ! -f "${CCTAG_CMAKE_DIR}/CCTagConfig.cmake" ]]; then
  echo "[ERROR] Expected CCTagConfig.cmake at: ${CCTAG_CMAKE_DIR}/CCTagConfig.cmake"
  exit 1
fi

echo "[OK] Installed CCTag into: ${INSTALL_PREFIX}"
echo "[OK] CCTag_DIR is: ${CCTAG_CMAKE_DIR}"