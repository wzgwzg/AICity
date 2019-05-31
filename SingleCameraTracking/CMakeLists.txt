cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")
include_directories(${PATH_TO_GCC4.8}/include)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PATH_TO_DEPENDENCIES}/eigen)
include_directories(${PATH_TO_DEPENDENCIES}/libboost)
include_directories(${PATH_TO_DEPENDENCIES}/gflags/build/include)
include_directories(${PATH_TO_CUDA}/cuda-7.5/include/)
include_directories(${PATH_TO_DEPENDENCIES}/openmpi-1.8/include)
include_directories(${PATH_TO_DEPENDENCIES}/caffe/include)
include_directories(./src
	./src/feature
	./src/matching
	./src/thirdPart
	./src/thirdPart/munkres
	./src/thirdPart/munkres/adapters)
set(CAFFE_LIBS
${PATH_TO_DEPENDENCIES}/caffe/build/lib/libcaffe.so)
set(GLOG_LIBS ${PATH_TO_DEPENDENCIES}/glog/lib/libglog.so)
set(CUDA_LIBS ${PATH_TO_CUDA}/cuda-7.5/lib64/libcudart.so.7.5 ${PATH_TO_CUDA}/cuda-7.5/lib64/libcublas.so.7.5 ${PATH_TO_CUDA}/cuda-7.5/lib64/libcurand.so.7.5)
set(CAFFE_DEPEND_LIBS ${PATH_TO_DEPENDENCIES}/caffe/lib)
set(LIBS ${OpenCV_LIBS} ${CAFFE_LIBS} ${GLOG_LIBS} ${CUDA_LIBS} 
         ${CAFFE_DEPENDENCY_LIB}/boost/lib/libboost_system.so)
aux_source_directory(. SRCS)

file(GLOB_RECURSE SRC_LIST src/*.cpp)
add_executable(run_tracker ${SRC_LIST})
target_link_libraries(run_tracker ${LIBS})
