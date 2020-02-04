add_executable(slater_integrator)
target_sources(slater_integrator PRIVATE main.cpp slater.cpp slater.h)
target_compile_features(slater_integrator PRIVATE cxx_std_11)
target_link_libraries(slater_integrator PRIVATE cubature
                                                BLAS_C
                                                LAPACK_C
                                                OpenMP::OpenMP_C)
install(TARGETS slater_integrator DESTINATION bin)
