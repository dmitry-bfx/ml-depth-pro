cmake_minimum_required(VERSION 3.20)
project(depth_pro_mlx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLX CONFIG REQUIRED)

add_library(depth_pro_mlx SHARED
    src/model.cpp
    src/depth_pro_mlx.cpp
)

target_include_directories(depth_pro_mlx
    PUBLIC include
    PRIVATE src
)

target_link_libraries(depth_pro_mlx PRIVATE mlx)

# Install
install(TARGETS depth_pro_mlx LIBRARY DESTINATION lib)
install(FILES include/depth_pro_mlx.h DESTINATION include)
