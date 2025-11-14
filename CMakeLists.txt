cmake_minimum_required(VERSION 3.16)

project(multiviewpatcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Additional Release optimizations
if(MSVC AND CMAKE_BUILD_TYPE STREQUAL Release)
  string(APPEND CMAKE_EXE_LINKER_FLAGS " /LTCG")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS " /LTCG")
  string(APPEND CMAKE_STATIC_LINKER_FLAGS " /LTCG")
endif()

add_subdirectory(spirv-tools EXCLUDE_FROM_ALL)
add_subdirectory(src)
add_subdirectory(example)
