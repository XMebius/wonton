cmake_minimum_required(VERSION 3.0)
project(Wonton)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(link_lib GTest::gtest glog::glog)
set(link_math_lib ${ARMADILLO_LIBRARIES})

file(GLOB SOURCES "src/*.cpp")
file(GLOB TEST_SOURCES "test/*.cpp")
add_executable(Wonton_1 main.cpp ${TEST_SOURCES} ${SOURCES})

target_link_libraries(Wonton_1 ${link_lib} ${link_math_lib})
target_include_directories(Wonton_1 PRIVATE ./include)