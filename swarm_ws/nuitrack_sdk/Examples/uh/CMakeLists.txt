cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME nuitrack_console_sample)
project(${PROJECT_NAME})

#add_definitions(-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0)
add_compile_options(-std=c++11)

set(NUITRACK_SDK_PATH ${CMAKE_CURRENT_LIST_DIR}/../..)

include_directories(
  ${NUITRACK_SDK_PATH}/Nuitrack/include   
  ${NUITRACK_SDK_PATH}/Nuitrack/include/middleware  
)

set(SOURCES
    src/main.cpp)
    
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(
  ${PROJECT_NAME}
  ${NUITRACK_SDK_PATH}/Nuitrack/lib/linux64/libnuitrack.so
  ${NUITRACK_SDK_PATH}/Nuitrack/lib/linux64/libmiddleware.so
)
