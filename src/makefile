slater: slater.cpp slater.h main.cpp
	g++ -o test.exe ../cubature/hcubature.c slater.cpp main.cpp -I /export/zimmerman/khoidang/local_mod/OpenBLAS/include/ -L/export/zimmerman/khoidang/local_mod/OpenBLAS/lib/ -lopenblas -lpthread
omp: slater.cpp slater.h main.cpp
	g++ -o test.exe ../cubature/hcubature.c slater.cpp main.cpp -I /export/zimmerman/khoidang/local_mod/OpenBLAS/include/ -L/export/zimmerman/khoidang/local_mod/OpenBLAS/lib/ -lopenblas -lpthread -fopenmp
clean:
	rm test.exe
