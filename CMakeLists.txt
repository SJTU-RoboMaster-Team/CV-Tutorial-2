cmake_minimum_required(VERSION 3.12)
project(homework2)

set(CMAKE_CXX_STANDARD 17)

find_package(fmt REQUIRED)
find_package(OpenCV 4 REQUIRED)

add_executable(homework2 main.cpp)
target_link_libraries(homework2 ${OpenCV_LIBS} fmt::fmt)
