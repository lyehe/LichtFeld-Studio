# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# External dependencies for new LichtFeld-Studio build (LibTorch-free)
# These are NOT used by the legacy build

# nvImageCodec - GPU-accelerated image codec library
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nvImageCodec/CMakeLists.txt")
    message(STATUS "Adding nvImageCodec from external/nvImageCodec")

    # Enable nvJPEG and nvJPEG2K extensions (the only ones we have)
    set(BUILD_NVJPEG_EXT ON CACHE BOOL "Build nvJPEG extension" FORCE)
    set(BUILD_NVJPEG2K_EXT ON CACHE BOOL "Build nvJPEG2000 extension" FORCE)

    # Create missing cmake template directory to avoid config error
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/template")
    # Create a minimal Config.cmake.in to satisfy nvImageCodec build
    file(WRITE "${CMAKE_SOURCE_DIR}/cmake/template/Config.cmake.in" "# Placeholder for nvImageCodec\n")

    add_subdirectory(nvImageCodec)
else()
    message(WARNING "nvImageCodec not found at external/nvImageCodec - GPU image loading will not be available")
endif()
