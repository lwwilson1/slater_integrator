cmake_minimum_required(VERSION 3.13)
project(slater_integrals VERSION 1.0 LANGUAGES C CXX)

set(default_build_type "Release")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_library(BLAS_C INTERFACE)
target_link_libraries(BLAS_C INTERFACE ${BLAS_LIBRARIES})
target_link_options(BLAS_C INTERFACE ${BLAS_LINKER_FLAGS})

add_library(LAPACK_C INTERFACE)
target_link_libraries(LAPACK_C INTERFACE ${LAPACK_LIBRARIES})
target_link_options(LAPACK_C INTERFACE ${LAPACK_LINKER_FLAGS})

add_subdirectory(cubature)
add_subdirectory(src)

