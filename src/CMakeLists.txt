cmake_minimum_required(VERSION 3.10)
project(ccc)

set(CMAKE_CXX_STANDARD 14)

add_library(ccc SHARED yolov4.cpp yolov4.h)
cmake_minimum_required(VERSION 3.10)

SET(common_dir /home/cmv/TensorRT-7.2.3.4/samples/common)
INCLUDE_DIRECTORIES(${common_dir})

SET(cuda_dir /usr/local/cuda-11.0/include)
INCLUDE_DIRECTORIES(${cuda_dir})

SET(cuda_lib /usr/local/cuda-11.0/lib64)
INCLUDE_DIRECTORIES(${cuda_lib})

find_package(OpenCV REQUIRED)
target_link_libraries(ccc ${OpenCV_LIBS})

SET(trt_dir /home/cmv/TensorRT-7.2.3.4/include)
INCLUDE_DIRECTORIES(${trt_dir})

find_library(LIBNVINFER nvinfer /home/cmv/TensorRT-7.2.3.4/targets/x86_64-linux-gnu/lib)
TARGET_LINK_LIBRARIES(ccc ${LIBNVINFER})

find_library(LIBCUDART cudart /usr/local/cuda-11.0/lib64/)
TARGET_LINK_LIBRARIES(ccc ${LIBCUDART})