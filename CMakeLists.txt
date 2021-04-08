cmake_minimum_required(VERSION 3.10)
project(fff)

set(CMAKE_CXX_STANDARD 14)
add_executable(fff main.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(fff ${OpenCV_LIBS})

link_directories(/home/cmv/CLionProjects/ccc/cmake-build-debug/)
target_link_libraries(fff libccc.so)
