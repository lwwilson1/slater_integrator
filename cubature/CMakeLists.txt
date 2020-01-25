cmake_minimum_required( VERSION 3.13 )
#updated this cmake

project( cubature )

add_library(cubature STATIC
    hcubature.c
    pcubature.c)

add_executable( htest test.c )
target_link_libraries( htest cubature m )

add_executable( ptest test.c )
target_link_libraries( ptest cubature m )
target_compile_definitions( ptest PRIVATE PCUBATURE=1 )

include(GNUInstallDirs)
install( TARGETS cubature LIBRARY DESTINATION lib
                          ARCHIVE DESTINATION lib)
install( FILES cubature.h DESTINATION include)

