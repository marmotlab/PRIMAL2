cmake_minimum_required(VERSION 3.14)
project(pybind)
add_subdirectory(pybind11)

include_directories("inc")
include_directories("src")
file(GLOB SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "./src/pybinding_CBS.cpp")
list(FILTER SOURCES EXCLUDE REGEX "./src/pybind.cpp")
# file(GLOB SOURCES "inc/*.h")
# set(SOURCE "./src/python_test.py")
# add_executable(test_pybind_exe ${SOURCE})

add_library(add STATIC ${SOURCES})
set_target_properties(add PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(cbs_py src/pybinding_CBS.cpp)
target_link_libraries(cbs_py PRIVATE add)

