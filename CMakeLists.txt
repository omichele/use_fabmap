project(use_fabmap)

cmake_minimum_required(VERSION 2.6)


## Cmake setup #################################################################

# tell cmake that the library goes in the library directory
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib)

# make a binary directory in the build directory
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

# tell cmake that the binaries goes in the binary directory
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

# Compiler warning level /W3 (msvc) or -Wall (gcc)
if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  # Update if necessary
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -pedantic")
endif()


# addtional compiler options
set(CMAKE_CXX_FLAGS
   "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native  -std=c++0x"
) 



## Required Packages ###########################################################

# OpenMP speedups
message(STATUS "")
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message("FOUND OPENMP")
endif(OPENMP_FOUND)

# Find OpenCV.
# If it's not found, set OpenCV_DIR to the directory with OpenCVConfig.cmake
if(WIN32)
    set(OpenCV_PATHS
        $ENV{OPENCV_HOME}
        $ENV{OPENCV_DIR}/../../
        C:/opencv/
        C:/OpenCV2.2/
        C:/OpenCV2.3/
        C:/OpenCV2.4/
        )
else() # Linux
    set(OpenCV_PATHS
        $ENV{OPENCV_HOME}/build
        /usr/local/share/OpenCV/
        /usr/share/OpenCV
        )
endif()
find_package(OpenCV REQUIRED NO_MODULE
    PATHS ${OpenCV_PATHS})
if(OpenCV_VERSION VERSION_GREATER "2.3.9")
    add_definitions(-DOPENCV2P4)
endif()


# FabMap
add_subdirectory(${PROJECT_SOURCE_DIR}/openfabmap)
include_directories(${PROJECT_SOURCE_DIR}/openfabmap/include)
# add_definitions("-DHAVE_FABMAP")
set(FABMAP_LIB openFABMAP )



## use_fabmap executable ####################################################

# Tell CMake where the headers are
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Tell the project where settings / doxygen / readme files are (for Qt Creator)
# file(GLOB SETTINGS_FILE "${CMAKE_CURRENT_SOURCE_DIR}/samples/settings.yml")
# file(GLOB DOXY_FILES "${CMAKE_CURRENT_SOURCE_DIR}"
#    ".travis.yml" "doxygen/*.dox" "doxygen/*.dox.*" "doxygen/*.xml")
# set(README_FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Copy the settings file across when building (not used for now)
#FILE(COPY ${SETTINGS_FILE} DESTINATION ${CMAKE_BINARY_DIR}/bin)

# Tell cmake about the binary
add_executable(use_fabmap ${CMAKE_SOURCE_DIR}/src/use_fabmap.cpp ${CMAKE_SOURCE_DIR}/src/FabMap.cpp)

add_executable(use_fabmap_video_st_lucia ${CMAKE_SOURCE_DIR}/src/use_fabmap_video_st_lucia.cpp ${CMAKE_SOURCE_DIR}/src/FabMap.cpp)

add_executable(use_fabmap_video_city_centre ${CMAKE_SOURCE_DIR}/src/use_fabmap_video_city_center.cpp ${CMAKE_SOURCE_DIR}/src/FabMap.cpp)

add_executable(use_fabmap_video_mono ${CMAKE_SOURCE_DIR}/src/use_fabmap_video_mono.cpp ${CMAKE_SOURCE_DIR}/src/FabMap.cpp)

# Tell openFABMAPcli to link against its required libs
target_link_libraries(use_fabmap ${FABMAP_LIB} ${OpenCV_LIBS} )

target_link_libraries(use_fabmap_video_st_lucia ${FABMAP_LIB} ${OpenCV_LIBS} )

target_link_libraries(use_fabmap_video_city_centre ${FABMAP_LIB} ${OpenCV_LIBS} )

target_link_libraries(use_fabmap_video_mono ${FABMAP_LIB} ${OpenCV_LIBS} )
