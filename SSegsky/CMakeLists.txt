cmake_minimum_required(VERSION 3.16)

project(segmented_range_graph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Main executable: cross-window edge builder
add_executable(crosswin
    cpp/src/main_crosswin.cpp
    cpp/src/dataset.cpp
    cpp/src/graph_io.cpp
    cpp/src/knn_state.cpp
    cpp/src/cross_window_search.cpp
    cpp/src/buffered_writer.cpp
)

target_include_directories(crosswin PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Query evaluation executable: range-filtered ANN search with dual backend (SLOT / TEXT)
add_executable(query_range_eval
    cpp/src/query_range_eval.cpp
)

target_include_directories(query_range_eval PRIVATE
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# OpenMP
find_package(OpenMP REQUIRED)
target_link_libraries(crosswin PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(query_range_eval PRIVATE OpenMP::OpenMP_CXX)

# Optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(crosswin PRIVATE
        -O3
        -Wall
        -Wextra
        -mavx2
        -mfma
    )
    target_compile_options(query_range_eval PRIVATE
        -O3
        -Wall
        -Wextra
        -mavx2
        -mfma
    )
endif()
