//Slater potential
// 2 or 3 center integrations

#define SHIFT_ZERO 1.e-30
#define MAX_EVAL 100000000
#define USE_OMP 0

#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <cblas.h>
#include <cmath>
#include <lapacke.h>

#include <math.h>
#include <vector>
using namespace std;
#include "../cubature/cubature.h"
#if USE_OMP
#include <omp.h>
#endif

#define PI 3.1415926535

//V(zeta,n,l,m;r,theta,phi) = N(zeta,n,l) * Zlm(theta,phi) * Inl(r)

//Inl(r) = r^(-l-1) * integral(r'^n+l+1 exp [ -zeta r' ] dr'
//           r^l *      integral(r'^n-l exp[-zeta r'] dr'


void print_square(int N, double* S)
{
  for (int n=0;n<N;n++)
  {
    for (int m=0;m<N;m++)
      printf(" %10.5f",S[n*N+m]);
    printf("\n");
  }
}

void print_square_ss(int N, double* S)
{
  for (int n=0;n<N;n++)
  {
    printf("  ");
    for (int m=0;m<N;m++)
      printf(" %8.5f",S[n*N+m]);
    printf("\n");
  }
}

void trans(double* Bt, double* B, int m, int n) {

  for (int i=0;i<m;i++)
  for (int j=0;j<n;j++)
    Bt[i*n+j] = B[j*m+i];

  return;
}

int Diagonalize(double* A, double* eigen, int size){

  if (size<1)
  {
    printf("  WARNING: cannot Diagonalize, size: %i \n",size);
    return 1;
  }

#define DSYEVX 1
 // printf(" in diagonalize call, size: %i \n",size);
 // printf(" in diagonalize: mkl_threads: %i \n",mkl_get_max_threads());

  int N = size;
  int LDA = size;
  double* EVal = eigen;

//borrowed from qchem liblas/diagon.C
    char JobZ = 'V', Range = 'A', UpLo = 'U';
    int IL = 1, IU = N;
    double AbsTol = 0.0, VL = 1.0, VU = -1.0;
    int NEValFound;

    double* EVec = new double[LDA*N];

    // Give dsyevx more work space than the minimum 8*N to improve performance
#if DSYEVX
    int LenWork = 32*N; //8*N min for dsyevx (was 32)
#else
    int LenWork = 1+6*N+2*N*N; //1+6*N+2*N*N min for dsyevd
#endif
    double* Work = new double[LenWork];

#if DSYEVX
    int LenIWork = 5*N; //5*N for dsyevx (was 5)
#else
    int LenIWork = 10*N; //3+5*N min for dsyevd
#endif
    int* IWork = new int[LenIWork];
    int* IFail = new int[N];

    int Info = 0;

#if USE_ACML
   dsyevx(JobZ, Range, UpLo, N, A, LDA, VL,
          VU, IL, IU, AbsTol, &NEValFound, EVal, EVec, LDA,
          IFail, &Info);
#else
#if DSYEVX
    dsyevx_(&JobZ, &Range, &UpLo, &N, A, &LDA, &VL, &VU, &IL, &IU, &AbsTol,
           &NEValFound, EVal, EVec, &LDA, Work, &LenWork, IWork, IFail, &Info);
#else
    dsyevd_(&JobZ, &UpLo, &N, A, &LDA, EVal, Work, &LenWork, IWork, &LenIWork, &Info);
#endif
#endif

#if 0
    if (Info != 0 && KillJob) {
      printf(" Info = %d\n",Info);
      QCrash("Call to dsyevx failed in Diagonalize");
    }
#endif

  int n_nonzero = 0;
  for (int i=0;i<size;i++)
  {
    //printf(" eigenvalue %i: %1.5f \n",i,eigen[i]);
    if (abs(eigen[i])>0.0001) n_nonzero++;
  }
  //printf(" found %i independent vectors \n",n_nonzero);

#if DSYEVX
  for (int i=0;i<size;i++)
  for (int j=0;j<size;j++)
    A[i*size+j] = EVec[i*size+j];
#endif

    delete [] EVec;
    delete [] Work;
    delete [] IWork;
    delete [] IFail;

  return 0;
}

int Invert(double* A, int m){

  if (m<1)
  {
    printf("  WARNING: cannot invert, size: %i \n",m);
    return 1;
  }

  int LenWork = 4*m;
  double* Work = new double[LenWork];

  int Info = 0;

  //printf(" LenWork: %i \n",LenWork);

  int* IPiv = new int[m];

  dgetrf_(&m,&m,A,&m,IPiv,&Info);
  if (Info!=0)
  {
    printf(" after dgetrf, Info error is: %i \n",Info);
    delete [] IPiv;
    delete [] Work;

    for (int i=0;i<m*m;i++) A[i] = 0.;
    for (int i=0;i<m;i++)
       A[i*m+i] = 1.;
 
    return 1;
  }

  dgetri_(&m,A,&m,IPiv,Work,&LenWork,&Info);
  if (Info!=0)
  {
    printf(" after invert, Info error is: %i \n",Info);
    printf(" A-1: \n");
    for (int i=0;i<m;i++)
    {
      for (int j=0;j<m;j++)
        printf(" %4.3f",A[i*m+j]);
      printf("\n");
    }
  }

  delete [] IPiv;
  delete [] Work;


  return 0;
}

int mat_root(double* A, int size)
{
  double* B = new double[size*size];
  for (int i=0;i<size*size;i++) B[i] = A[i]; 
  double* Beigen = new double[size];
  for (int i=0;i<size;i++) Beigen[i] = 0.;

  Diagonalize(B,Beigen,size);

  double* Bi = new double[size*size];
  //for (int i=0;i<size*size;i++) Bi[i] = B[i];

  trans(Bi,B,size,size);

  double* tmp = new double[size*size];
  for (int i=0;i<size*size;i++) tmp[i] = 0.;
  for (int i=0;i<size*size;i++) A[i] = 0.; 

  for (int i=0;i<size;i++)
  for (int j=0;j<size;j++)
    tmp[i*size+j] += Bi[i*size+j] * sqrt(Beigen[j]);
  for (int i=0;i<size;i++)
  for (int j=0;j<size;j++)
  for (int k=0;k<size;k++)
    A[i*size+j] += tmp[i*size+k] * B[k*size+j];


  delete [] B;
  delete [] Beigen;
  delete [] Bi;
  delete [] tmp;
  
  return 0;
}

int mat_root_inv(double* A, int size)
{
  double* B = new double[size*size];
  for (int i=0;i<size*size;i++) B[i] = A[i]; 
  double* Beigen = new double[size];
  for (int i=0;i<size;i++) Beigen[i] = 0.;

  Diagonalize(B,Beigen,size);

  double* Bi = new double[size*size];
  //for (int i=0;i<size*size;i++) Bi[i] = B[i];

  trans(Bi,B,size,size);

  double* tmp = new double[size*size]();
  for (int i=0;i<size*size;i++) A[i] = 0.; 

  for (int i=0;i<size;i++)
  for (int j=0;j<size;j++)
    tmp[i*size+j] += Bi[i*size+j] / sqrt(Beigen[j]);
  for (int i=0;i<size;i++)
  for (int j=0;j<size;j++)
  for (int k=0;k<size;k++)
    A[i*size+j] += tmp[i*size+k] * B[k*size+j];


  delete [] B;
  delete [] Beigen;
  delete [] Bi;
  delete [] tmp;
  
  return 0;
}

int mat_times_mat(double* C, double* A, double* B, int M, int N, int K)
{
  int LDA = K;
  int LDB = N;
  int LDC = N;

  double ALPHA = 1.0;
  double BETA = 0.0;
 
 //C := alpha*op( A )*op( B ) + beta*C (op means A or B, possibly transposed)
 //CBlas version
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

 //untested LIBLAS version
  char TA = 'N';
  char TB = 'N';
 // dgemm(&TA,&TB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);

#if 0
  printf(" printing C \n");
  for (int i=0;i<size;i++)
  {
    for (int j=0;j<size;j++)
      printf(" %6.3f",C[i*size+j]);
    printf("\n");
  }
#endif

  return 0;
}

int mat_times_mat(double* C, double* A, double* B, int size)
{
  int M = size;
  int N = size;
  int K = size;

  return mat_times_mat(C,A,B,M,N,K);
}

int mat_times_mat_at(double* C, double* A, double* B, int size)
{
  int M = size;
  int N = size;
  int K = size;

  int LDA = size;
  int LDB = size;
  int LDC = size;

  double ALPHA = 1.0;
  double BETA = 0.0;
 
 //C := alpha*op( A )*op( B ) + beta*C (op means A or B, possibly transposed)
 //CBlas version
  cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

 //untested LIBLAS version
  char TA = 'Y';
  char TB = 'N';
 // dgemm(&TA,&TB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);

#if 0
  printf(" printing C \n");
  for (int i=0;i<size;i++)
  {
    for (int j=0;j<size;j++)
      printf(" %6.3f",C[i*size+j]);
    printf("\n");
  }
#endif

  return 0;
}

int mat_times_mat_bt(double* C, double* A, double* B, int M, int N, int K)
{
  int LDA = K;
  int LDB = K;
  int LDC = N;

  double ALPHA = 1.0;
  double BETA = 0.0;
 
 //C := alpha*op( A )*op( B ) + beta*C (op means A or B, possibly transposed)
 //CBlas version
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

 //untested LIBLAS version
  char TA = 'N';
  char TB = 'N';
 // dgemm(&TA,&TB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);

  return 0;
}

int mat_times_mat_bt(double* C, double* A, double* B, int size)
{
  int M = size;
  int N = size;
  int K = size;

  int LDA = size;
  int LDB = size;
  int LDC = size;

  double ALPHA = 1.0;
  double BETA = 0.0;
 
 //C := alpha*op( A )*op( B ) + beta*C (op means A or B, possibly transposed)
 //CBlas version
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

 //untested LIBLAS version
  char TA = 'N';
  char TB = 'Y';
 // dgemm(&TA,&TB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);

#if 0
  printf(" printing C \n");
  for (int i=0;i<size;i++)
  {
    for (int j=0;j<size;j++)
      printf(" %6.3f",C[i*size+j]);
    printf("\n");
  }
#endif

  return 0;
}

int mat_times_mat_at_bt(double* C, double* A, double* B, int size)
{
  int M = size;
  int N = size;
  int K = size;

  int LDA = size;
  int LDB = size;
  int LDC = size;

  double ALPHA = 1.0;
  double BETA = 0.0;
 
 //C := alpha*op( A )*op( B ) + beta*C (op means A or B, possibly transposed)
 //CBlas version
  cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  char TA = 'Y';
  char TB = 'Y';
 //untested LIBLAS version
 // dgemm(&TA,&TB,&M,&N,&K,&ALPHA,A,&LDA,B,&LDB,&BETA,C,&LDC);

#if 0
  printf(" printing C \n");
  for (int i=0;i<size;i++)
  {
    for (int j=0;j<size;j++)
      printf(" %6.3f",C[i*size+j]);
    printf("\n");
  }
#endif

  return 0;
}

double Inr_1s(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double oz2 = 1./zeta/zeta;
  //double z3 = zeta*zeta*zeta;
  double oz3 = oz2/zeta;
  double ezr = exp(-zeta*r);
  double tz3r = 2.*oz3/r;

  double t1 = -ezr*(oz2 + tz3r);
  double t2 = tz3r;
  val = t1+t2;

  return val;
}

double Inr_2s(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double oz2 = 1./zeta/zeta;
  double roz2 = r*oz2;
  double oz3 = oz2/zeta;
  double soz4r = 6.*oz2*oz2/r;
  double ezr = exp(-zeta*r);

  double t1 = -ezr*(4.*oz3+roz2+soz4r);
  double t2 = soz4r;
  val = t1+t2;

  return val;
}


double Inr_2p(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double oz = 1./zeta;
  double oz2 = oz*oz;
  double oz3 = oz2*oz;
  double oz4 = oz2*oz2;
  double oz5 = oz3*oz2;
  double r2 = r*r;
  double ezr = exp(-zeta*r);
  double toz5r2 = 24.*oz5/r2;

  double t1 = -ezr*(toz5r2+24.*oz4/r+12.*oz3+3.*r*oz2);
  double t2 = toz5r2;
  val = t1+t2;

  return val;
}

double Inr_3s(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double oz2 = 1./zeta/zeta;
  double oz4 = oz2*oz2;
  double oz5r = oz4/r/zeta;
  double r2oz2 = r*r*oz2;
  double roz3 = oz2*r/zeta;
  double ezr = exp(-zeta*r);

  double t1 = -ezr*(24.*oz5r + 18.*oz4 + 6.*roz3 + r2oz2);
  double t2 = 24.*oz5r;
  val = t1+t2;

  return val;
}

double Inr_3p(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double r2 = r*r;
  double oz = 1./zeta;
  double oz2 = oz*oz;
  double oz3 = oz*oz2;
  double oz4 = oz2*oz2;
  double oz5r = oz2*oz3/r;
  double oz6r2 = oz3*oz3/r2;

  double ezr = exp(-zeta*r);

  //double t1 = -ezr*(120.*oz6 + 120.*roz5 + 60.*r2oz4 + (-2.*r + 20.*r3)*oz3 + -(2.*r2 + 5.*r4)*oz2 + (-r3 + r5)*oz);
  double t1 = -ezr*(120.*oz6r2 + 120.*oz5r + 60.*oz4 + 18.*r*oz3 + 5.*r2*oz2);
  double t2 = 120.*oz6r2;
  val = t1+t2;

  return val;
} 

double Inr_3d(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double r2 = r*r; double r3 = r2*r;

  double oz = 1./zeta;
  double oz2 = oz*oz;
  double oz3 = oz*oz2;
  double oz4 = oz2*oz2;
  double oz7 = oz3*oz4;

  double oz6r2 = oz2*oz4/r2;
  double oz5r = oz2*oz3/r;

  double ezr = exp(-zeta*r);
  double oz7r3 = oz7/r3;

  double t1 = -ezr*(720.*oz7r3 + 720.*oz6r2 + 360.*oz5r + 120.*oz4 + 30.*r*oz3 + 5.*r2*oz2);
  double t2 = 720.*oz7r3;
  val = t1+t2;

  return val;
}

//double Inr_4s(double zeta, double r)
//double Inr_4p(double zeta, double r)
//double Inr_4d(double zeta, double r)

double Inr_4f(double zeta, double r)
{
  double val;

  double zr = zeta*r;
  double r2 = r*r; double r3 = r*r2; double r4 = r2*r2; double r6 = r2*r4; double r7 = r4*r3; double r9 = r7*r2; double r10 = r7*r3;

  double oz = 1./oz;
  double oz2 = oz*oz;
  double oz3 = oz*oz2;
  double oz4 = oz2*oz2;
  double oz5 = oz2*oz3;

  double oz9r4 = oz4*oz5/r4;
  double oz8r3 = oz4*oz4/r3;
  double oz7r2 = oz4*oz3/r2;
  double oz6r = oz2*oz4/r;
  double roz4 = roz4;
  double r2oz3 = r2*oz3;
  double r3oz2 = r3*oz2;

  double ezr = exp(-zeta*r);
  double foz9r4 = 40320.*oz9r4;

  double t1 = -ezr*(foz9r4 + 40320.*oz8r3 + 20160.*oz7r2 + 6720.*oz6r + 1680.*oz5 + 336.*roz4 + 56.*r2oz3 + 7.*r3oz2);
  double t2 = foz9r4;
  val = t1+t2;

  return val;
}

int fact(int N)
{
  if (N==0) return 1;
  return N*fact(N-1);
}

double norm_sh(int l, int m)
{
  if (l==0)
    return 0.282094792;
  if (l==1)
    return 0.488602512;
  if (l==2)
  {
    if (m==0)
      return 0.315391565;
    if (m==2)
      return 0.54627422;
    return 1.092548431;
  }
  if (l==3)
  {
    if (m==0)
      return 0.37317633;
  }
  printf(" ERROR: norm not implemented for l=%i m=%i \n",l,m);
  exit(1);
  return 1.;
}

double norm_sv(int n, int l, int m, double zeta)
{
  double num = 4.*PI*pow(2.*zeta,n+0.5);
  double den = sqrt(fact(2*n))*(2.*l+1.);
  double val = num/den;
  val *= norm_sh(l,m);
  //printf(" norm_sv(n=%i l=%i m=%i zeta=%5.3f: %8.5f \n",n,l,m,zeta,val);
  return val;
}

double norm(int n, int l, int m, double zeta)
{
  double num = pow(2.*zeta,n+0.5);
  double den = sqrt(fact(2*n));

  double val = num/den;
  val *= norm_sh(l,m);
  //printf(" norm(n=%i l=%i m=%i zeta=%5.3f: %8.5f \n",n,l,m,zeta,val);
  return val;
}


double sh_1c(int l, int m, double x, double y, double z, double r)
{
  double val = 1.;

  if (l==1)
  {
    val = 1./r;
    if (m==0)      val *= x;
    else if (m==1) val *= y;
    else           val *= z;
  }
  else if (l==2)
  {
    val = 1./r/r;
    if (m==-2)      val *= x*y;
    else if (m==-1) val *= y*z;
    else if (m==0)  { val *= 3.*z*z; val -= 1; }
    else if (m==1)  val *= x*z;
    else if (m==2)  val *= x*x-y*y;
  }
  else if (l==3)
  {
    val = 1./r/r/r;
    if (m==-3) val *= (3.*x*x-y*y)*y;
    if (m==-2) val *= x*y*z;
    if (m==-1) val *= (4.*z*z-x*x-y*y)*y;
    if (m==0)  val *= (2.*z*z-3.*x*x-3.*y*y)*z;
    if (m==1)  val *= 1.;
   //not done
  }
  else if (l==4)
  {
    double or2 = 1./r/r;
    val = or2*or2;
    if (m==-4) val *= 1.;
    printf(" ERROR: l==4 not ready \n"); exit(1);
  }

  //printf(" sh_1c. xyz,r: %8.5g %8.5g %8.5g  %8.5g  val: %8.5g \n",x,y,z,r,val);

  return val;
}

double sh_2c(int l2, int m2, int l3, int m3, double x2, double y2, double z2, double x3, double y3, double z3, double r2, double r3)
{
  //printf(" sh_2c: %i %i %i %i xyz: %8.5g %8.5g %8.5g %8.5g %8.5g %8.5g r: %8.5g %8.5g \n",l2,m2,l3,m3,x2,y2,z2,x3,y3,z3,r2,r3);
  double val = sh_1c(l2,m2,x2,y2,z2,r2);
  val *= sh_1c(l3,m3,x3,y3,z3,r3);
  return val;
}

double slater_1s_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_1s(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  //double val = int_2c_1s_ns(n1,l1,zeta1,r1,n2,zeta2,r2);
  double val = I1*I2;
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  return val; 
}

double slater_2s_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_2s(zeta1,r1);
  //double I1 = 1.;
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  //double val = int_2c_2s_ns(n1,l1,zeta1,r1,n2,zeta2,r2);
  double val = I1*I2;
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  return val; 
}

double slater_2p_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_2p(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  //double val = int_2c_2p_ns(n1,l1,zeta1,r1,n2,zeta2,r2);
  double val = I1*I2;
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  //printf(" slater_2p_ns (%i/%i/%i) I1: %8.5f val: %12.10f \n",n1,l1,m1,I1,val);

  return val; 
}

double slater_3s_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_3s(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double val = I1*I2;
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  return val; 
}

double slater_3p_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_3p(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double val = I1*I2;
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  return val; 
}

double slater_3d_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_3d(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double val = I1*I2;
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  return val; 
}

double slater_4f_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = Inr_4f(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double val = I1*I2;
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_1c(l2,m2,x2,y2,z2,r2);

  return val; 
}

double slater_1s_ns_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;
  double C0 = par[18]; double C1 = 0.; double C2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;
  double x3 = k[0]-C0; double y3 = k[1]-C1; double z3 = k[2]-C2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;
  double r3 = sqrt(x3*x3+y3*y3+z3*z3)+SHIFT_ZERO;

  double I1 = Inr_1s(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double I3 = pow(r3,n3-1)*exp(-zeta3*r3);

  double val = I1*I2*I3;
  //double val = int_3c_1s_ns_ns(n1,l1,zeta1,r1,n2,zeta2,r2,n3,zeta3,r3);
  val *= sh_2c(l2,m2,l3,m3,x2,y2,z2,x3,y3,z3,r2,r3);

  return val; 
}

double slater_2s_ns_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  n1 = 2; l1 = 0; //given
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;
  double C0 = par[18]; double C1 = 0.; double C2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;
  double x3 = k[0]-C0; double y3 = k[1]-C1; double z3 = k[2]-C2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;
  double r3 = sqrt(x3*x3+y3*y3+z3*z3)+SHIFT_ZERO;

  double I1 = Inr_2s(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double I3 = pow(r3,n3-1)*exp(-zeta3*r3);

  double val = I1*I2*I3;
  //double val = int_3c_2s_ns_ns(n1,l1,zeta1,r1,n2,zeta2,r2,n3,zeta3,r3);
  val *= sh_2c(l2,m2,l3,m3,x2,y2,z2,x3,y3,z3,r2,r3);

  return val; 
}

double slater_2p_ns_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  n1 = 2; l1 = 1; //given, since 2p orbital
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;
  double C0 = par[18]; double C1 = 0.; double C2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;
  double x3 = k[0]-C0; double y3 = k[1]-C1; double z3 = k[2]-C2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;
  double r3 = sqrt(x3*x3+y3*y3+z3*z3)+SHIFT_ZERO;

  double I1 = Inr_2p(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double I3 = pow(r3,n3-1)*exp(-zeta3*r3);

  double val = I1*I2*I3;
  //double val = int_3c_2p_ns_ns(n1,l1,zeta1,r1,n2,zeta2,r2,n3,zeta3,r3);
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_2c(l2,m2,l3,m3,x2,y2,z2,x3,y3,z3,r2,r3);

  return val; 
}

double slater_3d_ns_ns(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  n1 = 3; l1 = 2; //given, since 3d orbital
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;
  double C0 = par[18]; double C1 = 0.; double C2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;
  double x3 = k[0]-C0; double y3 = k[1]-C1; double z3 = k[2]-C2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;
  double r3 = sqrt(x3*x3+y3*y3+z3*z3)+SHIFT_ZERO;

  double I1 = Inr_3d(zeta1,r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double I3 = pow(r3,n3-1)*exp(-zeta3*r3);

  double val = I1*I2*I3;
  //double val = int_3c_3d_ns_ns(n1,l1,zeta1,r1,n2,zeta2,r2,n3,zeta3,r3);
  val *= sh_1c(l1,m1,x1,y1,z1,r1);
  val *= sh_2c(l2,m2,l3,m3,x2,y2,z2,x3,y3,z3,r2,r3);

  return val; 
}

double ne1(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

 //location of nucleus
  double C0 = par[18]; double C1 = 0.; double C2 = 0.;
  double Zeff = par[21];

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;
  double x3 = k[0]-C0; double y3 = k[1]-C1; double z3 = k[2]-C2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;
  double Ra = sqrt(x3*x3+y3*y3+z3*z3)+SHIFT_ZERO;

  double I1 = pow(r1,n1-1)*exp(-zeta1*r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);

  double val = Zeff/Ra;
  val *= I1*I2;
  val *= sh_2c(l1,m1,l2,m2,x1,y1,z1,x2,y2,z2,r1,r2);

  return val;
}

double overlap1(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = pow(r1,n1-1)*exp(-zeta1*r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);

  double val = I1*I2;
  val *= sh_2c(l1,m1,l2,m2,x1,y1,z1,x2,y2,z2,r1,r2);

  return val;
}


double ke1(const double* k, size_t dim, void* params)
{
  double* par = (double*) params;

  int n1 = par[0]; int l1 = par[1]; int m1 = par[2];  double zeta1 = par[3];
  int n2 = par[4]; int l2 = par[5]; int m2 = par[6];  double zeta2 = par[7];
  int n3 = par[8]; int l3 = par[9]; int m3 = par[10]; double zeta3 = par[11];

  double A0 = par[12]; double A1 = 0.; double A2 = 0.;
  double B0 = par[15]; double B1 = 0.; double B2 = 0.;

  double x1 = k[0]-A0; double y1 = k[1]-A1; double z1 = k[2]-A2;
  double x2 = k[0]-B0; double y2 = k[1]-B1; double z2 = k[2]-B2;

  double r1 = sqrt(x1*x1+y1*y1+z1*z1)+SHIFT_ZERO;
  double r2 = sqrt(x2*x2+y2*y2+z2*z2)+SHIFT_ZERO;

  double I1 = pow(r1,n1-1)*exp(-zeta1*r1);
  double I2 = pow(r2,n2-1)*exp(-zeta2*r2);
  double rm1 = 1./r1; double rm2 = rm1*rm1; double zz1 = zeta1*zeta1;

  double val = (n1*(n1-1)-l1*(l1+1))*rm2-2*n1*zeta1*rm1+zz1;
  val *= I1*I2;
  val *= sh_2c(l1,m1,l2,m2,x1,y1,z1,x2,y2,z2,r1,r2);

  return val;
}

int ke_op(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = ke1(x,dim,data);
  return 0;
}

int ne_op(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = ne1(x,dim,data);
  return 0;
}

int overlap_op(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = overlap1(x,dim,data);
  return 0;
}

int slater_2c_1s(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_1s_ns(x,dim,data);
  return 0;
}

int slater_2c_2s(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_2s_ns(x,dim,data);
  return 0;
}

int slater_2c_2p(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_2p_ns(x,dim,data);
  return 0;
}

int slater_2c_3s(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  printf(" 2c_3s not available \n"); exit(1);
  retval[0] = slater_3s_ns(x,dim,data);
  return 0;
}

int slater_2c_3p(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  printf(" 2c_3p not available \n"); exit(1);
  retval[0] = slater_3p_ns(x,dim,data);
  return 0;
}

int slater_2c_3d(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_3d_ns(x,dim,data);
  return 0;
}

int slater_2c_4f(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_4f_ns(x,dim,data);
  return 0;
}

int slater_3c_1s(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_1s_ns_ns(x,dim,data);
  //retval[0] = slater_2s_ns_ns(x,dim,data);
  //retval[0] = slater_2p_ns_ns(x,dim,data);
  //retval[0] = slater_3d_ns_ns(x,dim,data);
  return 0;
}

int slater_3c_2s(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_2s_ns_ns(x,dim,data);
  return 0;
}

int slater_3c_2p(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_2p_ns_ns(x,dim,data);
  return 0;
}

int slater_3c_3s(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  printf(" slater_3c_3s not ready \n");
  //retval[0] = slater_3s_ns_ns(x,dim,data);
  return 0;
}

int slater_3c_3p(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  printf(" slater_3c_3p not ready \n");
  //retval[0] = slater_3p_ns_ns(x,dim,data);
  return 0;
}

int slater_3c_3d(unsigned dim, const double* x, void* data, unsigned fdim, double* retval)
{
  retval[0] = slater_3d_ns_ns(x,dim,data);
  return 0;
}

void display_results (char *title, double result, double error)
{
  printf ("%s ==================\n", title);
  printf ("result = % .6f\n", result);
  printf ("sigma  = % .6f\n", error);
  //printf ("exact  = % .6f\n", exact);
  //printf ("error  = % .6f = %.2g sigma\n", result - exact,
  //        fabs (result - exact) / error);
}

void integrate_2c(int type, int integrand_fdim, double* par, int dim, double* xl, double* xu, int maxEval, double tol, double& val, double& err)
{
  if (type==0) //1s
  {
    //printf(" integrate_2c 1s (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_1s, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==1) //2s
  {
    //printf(" integrate_2c 2s (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_2s, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==2) //2p
  {
    //printf(" integrate_2c 2p (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_2p, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==3) //3s
  {
    printf(" integrate_2c 3s (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_3s, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==4) //3p
  {
    printf(" integrate_2c 3p (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_3p, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==5) //3d
  {
    //printf(" integrate_2c 3d (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_3d, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

 //4s, 4p, 4d go here

  else if (type==9) //4f
  {
    printf(" integrate_2c 3d (%i %i) \n",(int)par[4],(int)par[5]);
    hcubature(integrand_fdim, slater_2c_4f, par,
        dim, xl, xu,
#if ABS_ERR
        maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else
  {
    printf(" this 2c integrand not available. type %i \n",type); exit(1);
  }

  return;
}

void integrate_3c(int type, int integrand_fdim, double* par, int dim, double* xl, double* xu, int maxEval, double tol, double& val, double& err)
{
  if (type==0) //1s
  {
    //printf(" integrate_3c 1s (%i %i/%i %i) \n",(int)par[4],(int)par[5],(int)par[8],(int)par[9]);
    hcubature(integrand_fdim, slater_3c_1s, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==1) //2s
  {
    //printf(" integrate_3c 2s (%i %i/%i %i) \n",(int)par[4],(int)par[5],(int)par[8],(int)par[9]);
    hcubature(integrand_fdim, slater_3c_2s, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==2) //2p
  {
    //printf(" integrate_3c 2p (%i %i/%i %i) \n",(int)par[4],(int)par[5],(int)par[8],(int)par[9]);
    hcubature(integrand_fdim, slater_3c_2p, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==3) //3s
  {
    //printf(" integrate_3c 3s (%i %i/%i %i) \n",(int)par[4],(int)par[5],(int)par[8],(int)par[9]);
    hcubature(integrand_fdim, slater_3c_3s, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==4) //3p
  {
    printf(" integrate_3c 3p (%i %i/%i %i) \n",(int)par[4],(int)par[5],(int)par[8],(int)par[9]);
    hcubature(integrand_fdim, slater_3c_3p, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==5) //3d
  {
    printf(" integrate_3c 3d (%i %i/%i %i) \n",(int)par[4],(int)par[5],(int)par[8],(int)par[9]);
    hcubature(integrand_fdim, slater_3c_3d, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }



#if 0
  else if (type==6) //4s
  {
    hcubature(integrand_fdim, slater_3c_4s, par,
       dim, xl, xu,
#if ABS_ERR
       maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
#else
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
#endif
  }

  else if (type==7) //4p
  {
    hcubature(integrand_fdim, slater_3c_4p, par,
       dim, xl, xu,
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  }

  else if (type==8) //4d
  {
    hcubature(integrand_fdim, slater_3c_4d, par,
       dim, xl, xu,
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  }

  else if (type==9) //4f
  {
    hcubature(integrand_fdim, slater_3c_4f, par,
       dim, xl, xu,
       maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  }
#endif

  else
  {
    printf(" this 3c integrand not available. type %i \n",type); exit(1);
  }

  return;
}

void integrate(int type, int integrand_fdim, double* par, int dim, double* xl, double* xu, int maxEval, double tol, double& val, double& err)
{
  if (type==0) //S
  {
    hcubature(integrand_fdim, overlap_op, par,
        dim, xl, xu,
    //    maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err); //relative tolerance
  }
  else if (type==1) //En
  {
    hcubature(integrand_fdim, ne_op, par,
        dim, xl, xu,
    //    maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  }
  else if (type==2) //KE
  {
    hcubature(integrand_fdim, ke_op, par,
        dim, xl, xu,
    //    maxEval, tol, 0, ERROR_INDIVIDUAL, &val, &err);
        maxEval, 0, tol, ERROR_INDIVIDUAL, &val, &err);
  }
  else
  {
    printf(" ERROR: cannot integrate that type (%i) \n",type); exit(1);
  }
  return;
}

void ao_to_mo(int N, double* jAO, double* jMO, double* jCA, double* tmp)
{
  mat_times_mat(tmp,jAO,jCA,N);
  mat_times_mat_at(jMO,jCA,tmp,N);

  return;
}

void mo_coeff_to_pao(int No, int N, double* jCA, double* Pao)
{
  for (int m=0;m<N*N;m++) Pao[m] = 0.;
  for (int i=0;i<No;i++)
  for (int m=0;m<N;m++)
  {
    for (int n=0;n<N;n++)
      Pao[m*N+n] += jCA[m*N+i]*jCA[n*N+i];
  }
  for (int m=0;m<N*N;m++)
    Pao[m] *= 2.;

  return;
}

double get_ehf(int No, int N, double* Pao, double* jHc, double* Fao)
{
  //printf("\n now computing E(HF) \n");
  int N2 = N*N;
  double Ehf = 0.;
  for (int m=0;m<N2;m++)
    Ehf += Pao[m]*(jHc[m]+Fao[m]);
  Ehf *= 0.5;
 
#if 0
 //this evaluates in MO basis
  int N2 = N*N;
  int N3 = N2*N;

  double Ehf = 0.;
  for (int m=0;m<No;m++)
    Ehf += jHc[m*N+m];
  for (int m=0;m<No;m++)
  for (int n=0;n<m;n++)
    Ehf += g[m*N3+m*N2+n*N+n];
  for (int m=0;m<No;m++)
  for (int n=0;n<No;n++)
    Ehf -= g[m*N3+n*N2+m*N+n];
#endif

  return Ehf;
}

void get_G(int N, double* Pao, double* g, double* G)
{
  int N2 = N*N;
  int N3 = N2*N;
  for (int m=0;m<N2;m++) G[m] = 0.;
  for (int s=0;s<N;s++)
  for (int l=0;l<N;l++)
  {
    double* g1 = &g[s*N3+l*N2];
    double p1 = Pao[s*N+l];
    for (int m=0;m<N;m++)
    for (int n=0;n<N;n++)
      G[m*N+n] += p1*g1[m*N+n];
    for (int m=0;m<N;m++)
    for (int n=0;n<N;n++)
      G[m*N+n] -= 0.5 * p1 * g[m*N3+l*N2+s*N+n];
  }
  return;
}

double hartree_fock(int No, int N, double* S, double* T, double* En, double* g)
{
  printf("\n\n\n Beginning HF code \n");

  double Ehf = 0.;

  int N2 = N*N;
  int N3 = N2*N;
  double* X = new double[N2]();
  double* Pao = new double[N2]();
  double* jHc = new double[N2];
  double* G = new double[N2];
  double* Fao = new double[N2];
  double* Fmo = new double[N2];
  double* Fp = new double[N2];
  double* Fe = new double[N];
  double* jCA = new double[N2]();
  double* tmp = new double[N2];
  double* tmp2 = new double[N2];

  for (int m=0;m<N2;m++) 
    X[m] = S[m];
  mat_root_inv(X,N);

  printf(" transformation matrix X: \n"); print_square(N,X);

  //mat_times_mat(tmp2,X,X,N);
  //printf(" transformation matrix X*X: \n"); print_square(N,tmp2);

  for (int m=0;m<N2;m++) tmp2[m] = S[m];
  Invert(tmp2,N);
  //printf(" mat_inv S: \n"); print_square(N,tmp2);

  if (N==2)
  {
   //some density in 1s orbitals
    Pao[0] = 1.; Pao[2] = 1.;
  }
  if (N==4)
  {
    jCA[0] = 1.;
    jCA[2*N+2] = 1.;
   //some density in 1s orbitals
    Pao[0] = 1.;
    Pao[2*N+2] = 1.;
  }

  for (int m=0;m<N2;m++)
    jHc[m] = En[m] + T[m];
  printf("\n Core Hamiltonian: \n"); print_square(N,jHc);

  double* dPao = new double[N2]();

  double dtol = 1.e-8;
  int maxsteps = 20;
  for (int ns=0;ns<maxsteps;ns++)
  {
    printf("\n\n Step %2i \n",ns);
    //printf(" current MO coeffs: \n"); print_square(N,jCA);

    get_G(N,Pao,g,G);
    //printf("\n G matrix: \n"); print_square(N,G);

    for (int m=0;m<N2;m++) Fao[m] = jHc[m] + G[m];
    //printf("\n Fock matrix in AO basis: \n"); print_square(N,Fao);

    Ehf = get_ehf(No,N,Pao,jHc,Fao);

    ao_to_mo(N,Fao,Fmo,jCA,tmp);
    //printf("\n Fock matrix in MO basis: \n"); print_square(N,Fmo);

    ao_to_mo(N,Fao,Fp,X,tmp);
    //printf("\n Fock matrix in orthogonal basis: \n"); print_square(N,Fp);

    Diagonalize(Fp,Fe,N);
    printf(" Fock eigenvalues:"); for (int m=0;m<N;m++) printf(" %8.5f",Fe[m]); printf("\n");
    //printf(" Fock eigenvectors: \n"); print_square(N,Fp);

    mat_times_mat_bt(jCA,X,Fp,N);
    //trans(jCA,tmp,N,N);
    //printf(" new MO coeffs: \n"); print_square(N,jCA);

#if 0
    printf("\n checking orthogonality of jCA \n");
    ao_to_mo(N,S,tmp2,jCA,tmp);
    print_square(N,tmp2);
#endif

    mo_coeff_to_pao(No,N,jCA,Pao);
    printf("\n new density: \n"); print_square(N,Pao);

    double dden = 0.; for (int m=0;m<N2;m++) { double d1 = Pao[m] - dPao[m]; dden += d1*d1; }
    printf("\n current E: %12.8f ",Ehf);
    printf(" density change: %12.10f \n",dden);
    if (dden<dtol) break;
    for (int m=0;m<N2;m++) dPao[m] = Pao[m];
  }

  printf("\n final MO coeffs: \n"); print_square(N,jCA);
  printf("\n final Fock matrix in MO basis: \n"); print_square(N,Fmo);


  delete [] dPao;

  delete [] X;
  delete [] Pao;
  delete [] jHc;
  delete [] G;
  delete [] Fao;
  delete [] Fmo;
  delete [] Fp;
  delete [] jCA;
  delete [] tmp;
  delete [] tmp2;

  return Ehf;
}

double nuclear_repulsion(int natoms, double* coords)
{
  double Enn = 0;
  if (natoms==2)
  {
    double x12 = coords[0] - coords[3];
    double y12 = coords[1] - coords[4];
    double z12 = coords[2] - coords[5];
    Enn += 1./sqrt(x12*x12+y12*y12+z12*z12);
  }
  else
  {
    printf(" ERROR: this function only for diatomics \n");
    exit(1);
  }
  return Enn;
}

void get_sign_xyz(int l, int m, int &ox, int &oy, int &oz)
{
  if (l==0) return;
  if (l==1)
  {
    if (m== 0) ox *= -1;
    if (m== 1) oy *= -1;
    if (m==-1) oz *= -1;
  }
  else if (l==2)
  {
   //sign doesn't change for m==0,2
    if (m==-2) { ox *= -1; oy *= -1; }
    if (m==-1) { oy *= -1; oz *= -1; }
    if (m== 1) { ox *= -1; oz *= -1; }
  }
  else if (l==3)
  {
   //xyz usually come in x2y2z2, so few terms here
    if (m==-3) { oy *= -1; }
    if (m==-2) { ox *= -1; oy *= -1; oz *= -1; }
    if (m==-1) { oy *= -1; }
    if (m== 0) { oz *= -1; }
    if (m== 1) { ox *= -1; }
    if (m== 2) { oz *= -1; }
    if (m== 3) { ox *= -1; }
  }
  else if (l==4)
  {
    printf("  get_sign_xyz not ready for l>=4 \n");
    exit(1);
  }
  return;
}

int check_2c_spherical(double x1, double x2, int l1, int l2, int m1, int m2)
{
  int integral_is_zero = 0;
  if (x1==x2) //if (par[12]==par[15])
  {
    if (l1!=l2) //if (par[1]!=par[5])
      integral_is_zero = 1;
    else if (m1!=m2) //else if (par[2]!=par[6])
      integral_is_zero = 1;
  }
  //if (integral_is_zero) printf(" found a zero \n");
  return integral_is_zero;
}

 //this function intended to check for multiple AOs on the same center, to resolve symmetry zeroes
int zero_check_3c(int n1, int l1, int m1, int n2, int l2, int m2, int n3, int l3, int m3, double x1, double x2, double x3)
{
  int ox = 1;
  int oy = 1;
  int oz = 1;
  int oa = 0;

  if (x1==x2 && x2==x3)
  {
    get_sign_xyz(l1,m1,ox,oy,oz);
    get_sign_xyz(l2,m2,ox,oy,oz);
    get_sign_xyz(l3,m3,ox,oy,oz);
    oa = 0; if (ox==-1) oa = 1; if (oy==-1) oa = 1; if (oz==-1) oa = 1;
    printf("  checking sign. (%i %i %i %i %i %i): %i %i %i -> %i \n",l1,m1,l2,m2,l3,m3,ox,oy,oz,oa);
  }

  return oa;
}

void compute_integrals(vector<vector<double> > basis, vector<vector<double> > basis_aux, double* coords, double* xl, double* xu, double* S, double* T, double* En, double* g, double tol)
{
  printf("\n computing ints \n");

  double val,err;
  int dim = 3;
  int integrand_fdim = 1;
  int maxEval = MAX_EVAL;

  int N = basis.size();
  int Naux = basis_aux.size();
  int Naux2 = Naux*Naux;
  int N2 = N*N;

  int nomp = 1;
#if USE_OMP
 #pragma omp parallel
  nomp = omp_get_num_threads();
  printf("\n found %i threads \n",nomp);
#endif

  int psize = 22;
  double* par0 = new double[psize*nomp]();
 //nuclear charge
  for (int m=0;m<nomp;m++)
    par0[psize*m+psize-1] = 1.;

  double Terr = 0.; double Enerr = 0.; double Serr = 0.;
#if USE_OMP
 #pragma omp parallel for private(val,err) reduction(+:Serr,Enerr,Terr)
#endif
  for (int n=0;n<N;n++)
  for (int m=0;m<=n;m++)
  {

#if USE_OMP
    int tid = omp_get_thread_num();
    double* par = &par0[tid*psize];
#else
    double* par = par0;
#endif

   //gather n,l,m,zeta
    par[0] = basis[n][0]; par[1] = basis[n][1]; par[2] = basis[n][2]; par[3] = basis[n][3];
    par[4] = basis[m][0]; par[5] = basis[m][1]; par[6] = basis[m][2]; par[7] = basis[m][3];

    par[12] = basis[n][5]; //X coordinate, diatomic only for the moment
    par[15] = basis[m][5];

    double norm12 = basis[n][4] * basis[m][4];

   //types: 0->overlap, 1->ne, 2->ke
    int overlap_ke_is_zero = check_2c_spherical(par[12],par[15],(int)par[1],(int)par[5],(int)par[2],(int)par[6]);
    if (!overlap_ke_is_zero)
      integrate(0,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
    else val = 0.;
    S[n*N+m] = val*norm12;
    Serr += err;
    //printf(" integrand(S[%i,%i]):  integral = %0.11g, est err = %g \n",n,m,val,err);

   //Electron-nuclear attraction
   //1st center
    par[18] = coords[0];
    integrate(1,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
    En[n*N+m] = -val*norm12;
    Enerr += err;
   //2nd center
    par[18] = coords[3];
    integrate(1,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
    En[n*N+m] -= val*norm12;
    Enerr += err;
    //printf(" integrand(En[%i,%i]): integral = %0.11g, est err = %g \n",n,m,val,err);

   //Kinetic Energy
    if (!overlap_ke_is_zero)
      integrate(2,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
    else val = 0.;
    T[n*N+m] = -0.5*val*norm12;
    Terr += err;
    //printf(" integrand(KE[%i,%i]): integral = %0.11g, est err = %g \n",n,m,val,err);
  }
  for (int n=0;n<N;n++) for (int m=0;m<n;m++) S[m*N+n]  = S[n*N+m];
  for (int n=0;n<N;n++) for (int m=0;m<n;m++) En[m*N+n] = En[n*N+m];
  for (int n=0;n<N;n++) for (int m=0;m<n;m++) T[m*N+n]  = T[n*N+m];
  
  double Nu = N*(N-1)/2 + N;
  Serr /= Nu; Terr /= Nu; Enerr /= Nu;

  printf("\n S: \n"); print_square(N,S);
  printf(" En: \n"); print_square(N,En);
  printf(" T: \n"); print_square(N,T);
  printf("\n");
  printf("  T/S/En errors: %g %g %g \n",Terr,Serr,Enerr);


  printf("\n working on (alpha|beta) integrals.");
  printf(" There are %3i of these \n",(Naux-1)*Naux/2+Naux);
 //auxiliary basis overlap
  double* A = new double[Naux2]();
  double Aerr = 0.;

#if USE_OMP
 #pragma omp parallel for private(val,err) reduction(+:Aerr)
#endif
  for (int m=0;m<Naux;m++)
  //for (int n=0;n<Naux;n++)
  for (int n=0;n<=m;n++)
  {

#if USE_OMP
    int tid = omp_get_thread_num();
    double* par = &par0[tid*psize];
    if (n==0 && tid==0) printf(".");
#else
    double* par = par0;
    if (n==0) printf(".");
#endif

    par[0] = basis_aux[m][0]; par[1] = basis_aux[m][1]; par[2] = basis_aux[m][2]; par[3] = basis_aux[m][3];
    par[4] = basis_aux[n][0]; par[5] = basis_aux[n][1]; par[6] = basis_aux[n][2]; par[7] = basis_aux[n][3];

    par[12] = basis_aux[m][5]; //X coordinate
    par[15] = basis_aux[n][5];

    double norm1v = norm_sv(par[0],par[1],par[2],par[3]);
    val = 0.;

    int integral_is_zero = check_2c_spherical(par[12],par[15],(int)par[1],(int)par[5],(int)par[2],(int)par[6]);
    if (!integral_is_zero)
    {
      if (par[0]==1) //1s function
        integrate_2c(0,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
      else if (par[0]==2)
      {
        if (par[1]==0) //2s
          integrate_2c(1,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
        else if (par[1]==1) //2p
          integrate_2c(2,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
      }
      else if (par[0]==3)
      {
        if (par[1]==0) //3s
          integrate_2c(3,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
        else if (par[1]==1) //3p
          integrate_2c(4,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
        else if (par[1]==2) //3d
          integrate_2c(5,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
      }
      else if (par[0]==4)
      {
        if (par[1]==3) //4f
          integrate_2c(9,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
      }
      //val *= norm1v * norm2;
      val *= norm1v * basis_aux[n][4];
    }
    A[m*Naux+n] = val;
    Aerr += err;

    //if (!integral_is_zero) printf(" found: %g \n",val);
    //if (integral_is_zero) printf(" zero expected. found: %g error: %g \n",val,err);
  }
  Aerr /= (Naux-1)*Naux/2 + Naux;
  printf("  average error: %g \n",Aerr); 
  printf("\n");
  for (int m=0;m<Naux;m++) for (int n=0;n<m;n++) A[n*Naux+m] = A[m*Naux+n];
  printf(" A: \n");
  for (int m=0;m<Naux;m++)
  {
    for (int n=0;n<Naux;n++)
      printf(" %8.5f",A[m*Naux+n]);
    printf("\n");
  }
  printf("\n");

  double* Ai = new double[Naux2];
  for (int m=0;m<Naux2;m++) Ai[m] = A[m];
  Invert(Ai,Naux);
#if 0
  printf(" A^-1: \n");
  for (int n=0;n<Naux;n++)
  {
    for (int m=0;m<Naux;m++)
      printf(" %8.3f",Ai[n*Naux+m]);
    printf("\n");
  }
  printf("\n");

  printf(" A^-1*A: \n");
  double* AiA = new double[Naux2]();
  mat_times_mat(AiA,Ai,A,Naux);
  for (int n=0;n<Naux;n++)
  {
    for (int m=0;m<Naux;m++)
      printf(" %8.5f",AiA[n*Naux+m]);
    printf("\n");
  }
  printf("\n");
#endif
  printf("\n");

 //Cm_ia
  printf("\n working on (alpha|mu nu) integrals");
  printf(" There are %3i of these \n",Naux*(N*(N-1)/2+N));
  int na = Naux;
  int nna = N*Naux;
  double* C = new double[Naux*N2]();
  double maxgerr = 0.;

#if USE_OMP
 #pragma omp parallel for private(val,err) reduction(max:maxgerr)
#endif
  for (int p=0;p<Naux;p++) 
  {

#if USE_OMP
    int tid = omp_get_thread_num();
    //printf("  i am thread %i \n",tid);
    double* par = &par0[tid*psize];
    if (tid==0) { printf("."); fflush(stdout); }
#else
    double* par = par0;
#endif

   //n,l,m,zeta; x position
    par[0] = basis_aux[p][0]; par[1] = basis_aux[p][1]; par[2] = basis_aux[p][2]; par[3] = basis_aux[p][3];
    par[12] = basis_aux[p][5]; 

    double norm1v = norm_sv(par[0],par[1],par[2],par[3]);

    for (int m=0;m<N;m++)
    for (int n=0;n<=m;n++)
    {
      val = 0.;

      par[4] = basis[m][0]; par[5] = basis[m][1]; par[6]  = basis[m][2]; par[7]  = basis[m][3];
      par[8] = basis[n][0]; par[9] = basis[n][1]; par[10] = basis[n][2]; par[11] = basis[n][3];
      par[15] = basis[m][5]; 
      par[18] = basis[n][5]; 

      int integral_is_zero = zero_check_3c(par[0],par[1],par[2],par[4],par[5],par[6],par[8],par[9],par[10],par[12],par[15],par[18]);
      //integral_is_zero = 0;
      if (!integral_is_zero)
      {
        double norm23 = basis[m][4]*basis[n][4];
        if (par[0]==1) //1s function
          integrate_3c(0,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
        else if (par[0]==2)
        {
          if (par[1]==0) //2s
            integrate_3c(1,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
          else if (par[1]==1) //2p
            integrate_3c(2,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
        }
        else if (par[0]==3)
        {
          if (par[1]==0) //3s
            integrate_3c(3,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
          else if (par[1]==1) //3p
            integrate_3c(4,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
          else if (par[1]==2) //3d
            integrate_3c(5,integrand_fdim,par,dim,xl,xu,maxEval,tol,val,err);
        }
        val *= norm1v*norm23; 
        C[m*nna+n*na+p] = val;
        //C[p*N2+m*N+n] = val;
        maxgerr = max(maxgerr,err);
        if (!integral_is_zero && fabs(val)<1.e-8) printf("   WARNING: computed a zero: %g. par: %i %i %i  %i %i %i  %i %i %i \n",val,(int)par[0],(int)par[1],(int)par[2],(int)par[4],(int)par[5],(int)par[6],(int)par[8],(int)par[9],(int)par[10]);
      }
      //if (integral_is_zero) printf(" zero expected. val: %g \n",val);
      //else printf(" val: %g \n",val);
      //printf(" integrand(PE): integral = %0.11g, est err = %g \n",val, err);
    }
    for (int m=0;m<N;m++)
    for (int n=0;n<m;n++)
      C[n*nna+m*na+p] = C[m*nna+n*na+p];
    //  C[p*N2+n*N+m] = C[p*N2+m*N+n];
  }
  printf("\n");
  printf("  largest error: %g \n",maxgerr); 
  fflush(stdout);

  double* AC = new double[Naux*N2]();
  int nanan2 = Naux2+N2;
  double* tmp = new double[nomp*nanan2];
#if 1

#if USE_OMP
 #pragma omp parallel for
#endif
  for (int m=0;m<N;m++)
  {
    int tid = 0;
#if USE_OMP
    tid = omp_get_thread_num();
#endif
    double* tmp1 = &tmp[tid*nanan2];
    mat_times_mat(tmp1,&C[m*nna],Ai,N,na,na);
    trans(&AC[m*nna],tmp1,na,N);
  }

#elif 1
  for (int m=0;m<N;m++)
  {
    double* AC1 = &AC[m*nna];
    double* C1 = &C[m*nna];
    for (int n=0;n<N;n++)
    for (int p=0;p<Naux;p++)
    for (int q=0;q<Naux;q++)
      AC1[n*na+p] += Ai[p*Naux+q] * C1[n*na+q];
  }
#else
 //crude contraction
  for (int m=0;m<N;m++)
  for (int n=0;n<N;n++)
  for (int p=0;p<Naux;p++)
  for (int q=0;q<Naux;q++)
    AC[m*nna+n*na+p] += Ai[p*Naux+q] * C[m*nna+n*na+q];
    //AC[p*N2+m*N+n] += Ai[p*Naux+q] * C[q*N2+m*N+n];
#endif

  int N3 = N2*N;

#if 1
  for (int m=0;m<N;m++)
  for (int l=0;l<N;l++)
  {
    double* C1 = &C[m*nna];
    double* AC1 = &AC[l*nna];
    mat_times_mat(tmp,C1,AC1,N,N,na);
    for (int n=0;n<N;n++)
    for (int s=0;s<N;s++)
      g[m*N3+n*N2+l*N+s] = tmp[n*N+s];
  }
#else
 //crude contraction
  for (int m=0;m<N;m++)
  for (int n=0;n<N;n++)
  {
    for (int l=0;l<N;l++)
    for (int s=0;s<N;s++)
    {
      for (int p=0;p<Naux;p++)
        g[m*N3+n*N2+l*N+s] += C[m*nna+n*na+p] * AC[l*nna+p*N+s];
        //g[m*N3+n*N2+l*N+s] += C[m*nna+n*na+p] * AC[l*nna+s*na+p];
        //g[m*N3+n*N2+l*N+s] += C[p*N2+m*N+n] * AC[p*N2+l*N+s];
    }
  }
#endif

  printf("\n");
  for (int m=0;m<N3*N;m++)
  if (fabs(g[m])>100.)
  {
    printf("  g matrix element huge, exiting \n");
    printf(" g(%2i): %10.8f \n",m,g[m]);
    exit(1);
  }
  printf("\n printing gmnls \n");
  for (int m=0;m<N;m++)
  for (int n=0;n<=m;n++)
  {
    printf("  g%i%i:\n",m,n);
    print_square_ss(N,&g[m*N3+n*N2]);
  }

  delete [] A;
  delete [] Ai;
  delete [] C;
  delete [] par0;

  return;
}

void add_p(vector<vector<double> > &basis_aux, vector<double> ao1, double zeta)
{
 //2px
  ao1[0] = 2; ao1[1] = 1; ao1[2] = 0;
  ao1[3] = zeta;
  ao1[4] = norm(2,1,0,zeta);
  basis_aux.push_back(ao1);

 //2py
  ao1[2] = 1; 
  basis_aux.push_back(ao1);
 //2pz
  ao1[2] = -1;
  basis_aux.push_back(ao1);

  return;
}
