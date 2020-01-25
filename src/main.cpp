#include <stdlib.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cblas.h>
#include <lapacke.h>

#include <math.h>
#include <vector>
using namespace std;
#include "../cubature/cubature.h"
#if USE_OMP
#include <omp.h>
#endif
#include "slater.h"

int main (void)
{
  int dim = 3;
  double xmax = 35.;
  double xl[3] = { -xmax, -xmax, -xmax };
  double xu[3] = { xmax, xmax, xmax };

 //geometry of diatomic
  double* coords = new double[6]();
  coords[3] = 1.4;
  double Enn = nuclear_repulsion(2,coords);

  double ezrmax = exp(-xmax);
  printf("  exp(-zr) at r=max: %5.3e \n",ezrmax);



  vector<vector<double> > basis;

 //DZ basis set for H2
  {
    double zeta1s1 = 1.28;
    double zeta1s2 = 0.76;
    double zeta1s3 = 0.69; //use with 1.58, 0.92 for TZ
    double zeta2p1 = 1.25;

   //2x 1s (center A)
    vector<double> ao1;
    ao1.push_back(1); ao1.push_back(0); ao1.push_back(0);
    ao1.push_back(zeta1s1);
    ao1.push_back(norm(1,0,0,zeta1s1));
    ao1.push_back(coords[0]); ao1.push_back(coords[1]); ao1.push_back(coords[2]);
    basis.push_back(ao1);

    ao1[3] = zeta1s2; 
    ao1[4] = norm(1,0,0,zeta1s2);
    basis.push_back(ao1);

    ao1[0] = 2; ao1[1] = 1;
    ao1[3] = zeta2p1;
    ao1[4] = norm(2,1,0,zeta2p1);
    basis.push_back(ao1);

   //2x 1s (center B)
    vector<double> ao2;
    ao2.push_back(1); ao2.push_back(0); ao2.push_back(0);
    ao2.push_back(zeta1s1);
    ao2.push_back(norm(1,0,0,zeta1s1));
    ao2.push_back(coords[3]); ao2.push_back(coords[4]); ao2.push_back(coords[5]);
    basis.push_back(ao2);

    ao2[3] = zeta1s2;
    ao2[4] = norm(1,0,0,zeta1s2);
    basis.push_back(ao2);

    ao2[0] = 2; ao2[1] = 1;
    ao2[3] = zeta2p1;
    ao2[4] = norm(2,1,0,zeta2p1);
    basis.push_back(ao2);
  }

  int N = basis.size();
  int N2 = N*N;
  printf("  found %i basis functions \n",N);
  for (int n=0;n<N;n++)
    printf(" basis(%i): %i %i %i %5.3f norm: %5.3f \n",n,(int)basis[n][0],(int)basis[n][1],(int)basis[n][2],basis[n][3],basis[n][4]);



  vector<vector<double> > basis_aux;

 //auxiliary basis set for H2
  {
    double zeta1s1 = 3.16;
    double zeta1s2 = 2.09;
    double zeta1s3 = 1.38;
    double zeta2s1 = 1.50;
    double zeta2p1 = 4.00;
    double zeta2p2 = 2.65;
    double zeta2p3 = 1.75;
    double zeta3d1 = 4.00;
    double zeta3d2 = 2.50;

   //4x 1s (center A)
    vector<double> ao1;
    ao1.push_back(1); ao1.push_back(0); ao1.push_back(0);
    ao1.push_back(zeta1s1);
    ao1.push_back(norm(1,0,0,zeta1s1));
    ao1.push_back(coords[0]); ao1.push_back(coords[1]); ao1.push_back(coords[2]);
    basis_aux.push_back(ao1);

    ao1[3] = zeta1s2; 
    ao1[4] = norm(1,0,0,zeta1s2);
    basis_aux.push_back(ao1);

    ao1[3] = zeta1s3; 
    ao1[4] = norm(1,0,0,zeta1s3);
    basis_aux.push_back(ao1);

   //2s
    ao1[0] = 2;
    ao1[3] = zeta2s1; 
    ao1[4] = norm(2,0,0,zeta2s1);
    basis_aux.push_back(ao1);

    add_p(basis_aux,ao1,zeta2p1);
    add_p(basis_aux,ao1,zeta2p2);
    add_p(basis_aux,ao1,zeta2p3);

   //3dxy
    ao1[0] = 3; ao1[1] = 2; ao1[2] = -2;
    ao1[3] = zeta3d1;
    ao1[4] = norm(3,2,-2,zeta3d1);
    basis_aux.push_back(ao1);

   //4x 1s (center B)
    vector<double> ao2;
    ao2.push_back(1); ao2.push_back(0); ao2.push_back(0);
    ao2.push_back(zeta1s1);
    ao2.push_back(norm(1,0,0,zeta1s1));
    ao2.push_back(coords[3]); ao2.push_back(coords[4]); ao2.push_back(coords[5]);
    basis_aux.push_back(ao2);

    ao2[3] = zeta1s2;
    ao2[4] = norm(1,0,0,zeta1s2);
    basis_aux.push_back(ao2);

    ao2[3] = zeta1s3;
    ao2[4] = norm(1,0,0,zeta1s3);
    basis_aux.push_back(ao2);

   //2s
    ao2[0] = 2;
    ao2[3] = zeta2s1;
    ao2[4] = norm(2,0,0,zeta2s1);
    basis_aux.push_back(ao2);

    add_p(basis_aux,ao2,zeta2p1);
    add_p(basis_aux,ao2,zeta2p2);
    add_p(basis_aux,ao2,zeta2p3);

   //3dxy
    ao2[0] = 3; ao2[1] = 2; ao2[2] = -2;
    ao2[3] = zeta3d1;
    ao2[4] = norm(3,2,-2,zeta3d1);
    basis_aux.push_back(ao2);
  }

  int Naux = basis_aux.size();
  printf("  found %i auxiliary basis functions \n",Naux);
  for (int n=0;n<Naux;n++)
    printf(" basis_aux(%2i): %2i %2i %2i %5.3f norm: %5.3f \n",n,(int)basis_aux[n][0],(int)basis_aux[n][1],(int)basis_aux[n][2],basis_aux[n][3],basis_aux[n][4]);


  //generate 1e integrals
  double* T = new double[N2]();
  double* En = new double[N2]();
  double* g = new double[N2*N2]();
  double* S = new double[N2]();

  double tol = 1.e-4;
  compute_integrals(basis,basis_aux,coords,xl,xu,S,T,En,g,tol);

  int No = 1;
  double Ehf = hartree_fock(No,N,S,T,En,g) + Enn;
  printf("\n final HF energy: %12.8f w/Enn: %12.8f \n",Ehf-Enn,Ehf);


 //clean up
  delete [] T;
  delete [] En;
  delete [] g;
  delete [] S;

  delete [] coords;

  return 0;
}
