cmake_minimum_required(VERSION 2.8.3)
project(reid_thermal)

find_package(catkin REQUIRED COMPONENTS 
    cv_bridge 
    geometry_msgs 
    image_transport 
    message_filters 
    message_generation
    roscpp 
    sensor_msgs 
    std_msgs 
    visualization_msgs
    dynamic_reconfigure
    )

find_package(Boost REQUIRED COMPONENTS thread)

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
endif()

#######################################
## Declare ROS messages and services ##
#######################################

# add_message_files(
#  FILES
#  UpperBodyDetector.msg
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs
# )

#generate_dynamic_reconfigure_options(
#  cfg/reidentifier.cfg
#)


###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS 
    cv_bridge 
    geometry_msgs 
    image_transport 
    message_filters 
    roscpp 
    sensor_msgs 
    std_msgs 
    visualization_msgs
    dynamic_reconfigure
)

###########
## Build ##
###########

add_compile_options(-std=c++11)

include_directories( include
  ${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

add_executable(reid_thermal 
    src/main.cpp 
    src/svm.cpp
) 

add_executable(train
    src/train.cpp 
    src/svm.cpp
) 

add_dependencies(reid_thermal irimager
  irimageprocessing
  ircore ${reid_thermal_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#add_dependencies(reid_thermal ${PROJECT_NAME}_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
add_dependencies(train irimager
  irimageprocessing
  ircore ${reid_thermal_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(reid_thermal irimager
  irimageprocessing
  ircore
  udev
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(train irimager
  irimageprocessing
  ircore
  udev
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)
#############
## Install ##
#############

install(PROGRAMS
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(TARGETS reid_thermal
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(TARGETS train
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

