cmake_minimum_required( VERSION 2.8 )
project( vo1 )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${G2O_INCLUDE_DIRS}
        ${CSPARSE_INCLUDE_DIR}
        ${CERES_INCLUDE_DIRS}
        #${Sophus_INCLUDE_DIRS}
        "/usr/local/include/Eigen"
)

find_package( OpenCV 3.1 REQUIRED )
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2 
find_package( G2O REQUIRED )
#include_directories(
#        ${G2O_INCLUDE_DIRS}
#)
find_package(CSparse REQUIRED )
find_package(Sophus REQUIRED)
find_package( Ceres REQUIRED )
find_package(Eigen3 REQUIRED)
#message()



#add_executable( hand-eye pose_estimation.cpp)  # hand-eye_calibration.cpp
#target_link_libraries( hand-eye
#        ${OpenCV_LIBS}
#        ${Sophus_LIBRARIES}
#        ${CERES_LIBRARIES}
#        )

add_executable( hand-eye hand-eye_calibration.cpp)  # hand-eye_calibration.cpp     pose_estimation.cpp
target_link_libraries( hand-eye
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
        ${CERES_LIBRARIES}
        )

#add_executable( feature_extraction feature_extraction.cpp  )
##target_link_libraries( feature_extraction
##        ${OpenCV_LIBS}
##        )



