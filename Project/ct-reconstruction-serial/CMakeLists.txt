cmake_minimum_required(VERSION 2.8)

project(ct-reconstruction-serial)

find_package(OpenCV)

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
