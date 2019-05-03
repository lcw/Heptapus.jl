#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

// to build on Titan V:
//   nvcc -arch=sm_70 --ptxas-options=-v -o vanilladeriv vanilladeriv.cu;

#ifdef USE_DOUBLE
#define dfloat double
#else
#define dfloat float
#endif

#ifndef POLYNOMIAL_ORDER
#define POLYNOMIAL_ORDER 4
#endif

template <int Nq, int Np>
  __global__ void volumerhs(dfloat * __restrict__ rhs,
                            const dfloat * __restrict__ Q,
                            const int nelem){

  __shared__ dfloat s_F[Nq][Nq];
  dfloat r_rhsR[Nq];

  int e = blockIdx.x;
  int j = threadIdx.y;
  int i = threadIdx.x;

#pragma unroll Nq
  for(int k=0;k<Nq;++k) r_rhsR[k] = 0;

#pragma unroll Nq
  for(int k=0;k<Nq;++k){

    __syncthreads();

    int qid = i + j*Nq + k*Nq*Nq + e*Np;

    s_F[i][j] = Q[qid];

    __syncthreads();

#pragma unroll Nq
    for(int n=0;n<Nq;++n){
      r_rhsR[k] += s_F[n][j];
      r_rhsR[k] += s_F[n][i];
      r_rhsR[k] += s_F[j][n];
      r_rhsR[k] += s_F[i][n];
    }
  }

#pragma unroll Nq
  for(int k=0;k<Nq;++k){
    int qid = i + j*Nq + k*Nq*Nq + e*Np;
    rhs[qid] += r_rhsR[k];
  }
}

void randArray(int N, dfloat base, dfloat range, dfloat **q, dfloat **c_q){

  *q = (dfloat*) calloc(N, sizeof(dfloat));
  cudaMalloc(c_q, N*sizeof(dfloat));

  for(int n=0;n<N;++n){
    q[0][n] = base + drand48()*range;
  }

  cudaMemcpy(c_q[0], q[0], N*sizeof(dfloat), cudaMemcpyHostToDevice);

}

int main(int argc, char **argv){

  srand48(1234);

  const int N = POLYNOMIAL_ORDER;
  const int nelem = 4000;

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;

  const int Ntotal = Np*nelem;

  dfloat *Q, *c_Q;
  randArray(Ntotal, 0., 1., &Q, &c_Q);

  cudaMemcpy(c_Q, Q, nelem*Np*sizeof(dfloat), cudaMemcpyHostToDevice);

  dfloat *rhs, *c_rhs;

  srand48(1234);
  randArray(Ntotal, 1., 1., &rhs, &c_rhs);

  dim3 G(nelem,1,1);
  dim3 B2(Nq,Nq,Nq);
  dim3 B3(Nq,Nq,1);

  volumerhs<Nq, Np> <<< G, B3 >>> (c_rhs, c_Q, nelem);

  cudaDeviceSynchronize();

  exit(0);
  return 0;
}
