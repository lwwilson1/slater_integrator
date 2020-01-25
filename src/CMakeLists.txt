add_executable(slater_integrator)
target_sources(slater_integrator PRIVATE main.cpp slater.cpp slater.h)
target_link_libraries(slater_integrator PRIVATE cubature
                                                BLAS_C
                                                OpenMP::OpenMP_C)
install(TARGETS slater_integrator DESTINATION bin)
