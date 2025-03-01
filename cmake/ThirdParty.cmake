# >>> 导入第三方库 >>>

# 导入Torch库
set(Torch_DIR $ENV{Torch_DIR})
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

# 导入Eigen库
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# 导入yaml-cpp库

set(Ceres_DIR /usr/local/include/Ceres-2.1.0/lib/cmake/Ceres)
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


# 导入PCL库
if (DEFINED ENV{PCL_DIR})
  set(PCL_DIR $ENV{PCL_DIR})
  INFO_LOG("Using Custom PCL_DIR：${PCL_DIR}")
endif ()

find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})
INFO_LOG("PCL_VERSION is ${PCL_VERSION}")

# 导入OpenCV库
find_package(OpenCV 4 REQUIRED QUIET)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(DBoW3 REQUIRED)
include_directories(${DBoW3_INCLUDE_DIRS})

# 导入CUDA、NVCC、TensorRT库
include(cmake/TensorRT.cmake)