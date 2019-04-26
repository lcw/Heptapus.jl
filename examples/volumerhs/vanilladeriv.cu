#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>

// to build on Titan V:
//   nvcc -arch=sm_70 --ptxas-options=-v -o vanilladeriv vanilladeriv.cu;
//
// to run with 512 elements:
//   ./vanilladeriv 512

#ifdef USE_DOUBLE
#define dfloat double
#else
#define dfloat float
#endif

#ifndef POLYNOMIAL_ORDER
#define POLYNOMIAL_ORDER 4
#endif

// note the order of the fields below is also assumed in the code.
const int _nstate = 5;

const int _R = 0, _U  = 1, _V  = 2, _W = 3, _E = 4;

const int _nvgeo = 14;
const int _XIx   = 0;
const int _ETAx  = 1;
const int _ZETAx = 2;
const int _XIy   = 3;
const int _ETAy  = 4;
const int _ZETAy = 5;
const int _XIz   = 6;
const int _ETAz  = 7;
const int _ZETAz = 8;
const int _MJ    = 9;
const int _MJI   = 10;
const int _x     = 11;
const int _y     = 12;
const int _z     = 13;

#define grav  ((dfloat) 9.81)
#define R_d   ((dfloat)287.0024093890231)
#define cp_d  ((dfloat)1004.5084328615809)
#define cv_d  ((dfloat)717.5060234725578)
#define gamma_d  ((dfloat)1.4)
#define gdm1      ((dfloat)0.4)
  

// Volume RHS for 3-D
template <int Nq, int Np,  int nmoist, int ntrace, int nvar>
  __global__ void volumerhs_v2(dfloat * __restrict__ rhs,
			       const dfloat * __restrict__ Q,
			       const dfloat * __restrict__ vgeo,
			       const dfloat gravity,
			       const dfloat * __restrict__ D,
			       const int nelem){
  
  __shared__ dfloat s_D[Nq][Nq];
  __shared__ dfloat s_F[Nq][Nq][Nq][_nstate];
  __shared__ dfloat s_G[Nq][Nq][Nq][_nstate];
  __shared__ dfloat s_H[Nq][Nq][Nq][_nstate];

  int e = blockIdx.x;
  int k = threadIdx.z;
  int j = threadIdx.y;
  int i = threadIdx.x;
  
  if(k == 0)
    s_D[j][i] = D[j*Nq+i];
  
  // Load values will need into registers
  int gid = i + j*Nq + k*Nq*Nq + e*Np*_nvgeo;

  dfloat MJ = vgeo[gid + _MJ*Np];
  dfloat XIx = vgeo[gid + _XIx*Np];
  dfloat XIy = vgeo[gid + _XIy*Np];
  dfloat XIz = vgeo[gid + _XIz*Np];
  dfloat ETAx = vgeo[gid + _ETAx*Np];
  dfloat ETAy = vgeo[gid + _ETAy*Np];
  dfloat ETAz = vgeo[gid + _ETAz*Np];
  dfloat ZETAx = vgeo[gid + _ZETAx*Np];
  dfloat ZETAy = vgeo[gid + _ZETAy*Np];
  dfloat ZETAz = vgeo[gid + _ZETAz*Np];
  dfloat  z = vgeo[gid +  _z*Np];        
  
  int qid = i + j*Nq + k*Nq*Nq + e*Np*nvar;
  
  dfloat R = Q[qid + _R*Np];
  dfloat U = Q[qid + _U*Np];
  dfloat V = Q[qid + _V*Np];
  dfloat W = Q[qid + _W*Np];
  dfloat E = Q[qid + _E*Np];

  dfloat P = gdm1*(E - (U*U + V*V + W*W)/(2*R) - R*gravity*z);

  dfloat Rinv = 1 / R;  

  dfloat fluxR_x = U;
  dfloat fluxU_x = Rinv * U * U + P;
  dfloat fluxV_x = Rinv * U * V;
  dfloat fluxW_x = Rinv * U * W;
  dfloat fluxE_x = Rinv * U * (E + P);
  
  dfloat fluxR_y = V;
  dfloat fluxU_y = Rinv * V * U;
  dfloat fluxV_y = Rinv * V * V + P;
  dfloat fluxW_y = Rinv * V * W;
  dfloat fluxE_y = Rinv * V * (E + P);
  
  dfloat fluxR_z = W;
  dfloat fluxU_z = Rinv * W * U;
  dfloat fluxV_z = Rinv * W * V;
  dfloat fluxW_z = Rinv * W * W + P;
  dfloat fluxE_z = Rinv * W * (E + P);

  s_F[i][j][k][_R] = MJ * (XIx * fluxR_x + XIy * fluxR_y + XIz * fluxR_z);
  s_F[i][j][k][_U] = MJ * (XIx * fluxU_x + XIy * fluxU_y + XIz * fluxU_z);
  s_F[i][j][k][_V] = MJ * (XIx * fluxV_x + XIy * fluxV_y + XIz * fluxV_z);
  s_F[i][j][k][_W] = MJ * (XIx * fluxW_x + XIy * fluxW_y + XIz * fluxW_z);
  s_F[i][j][k][_E] = MJ * (XIx * fluxE_x + XIy * fluxE_y + XIz * fluxE_z);

  s_G[i][j][k][_R] = MJ * (ETAx * fluxR_x + ETAy * fluxR_y + ETAz * fluxR_z);
  s_G[i][j][k][_U] = MJ * (ETAx * fluxU_x + ETAy * fluxU_y + ETAz * fluxU_z);
  s_G[i][j][k][_V] = MJ * (ETAx * fluxV_x + ETAy * fluxV_y + ETAz * fluxV_z);
  s_G[i][j][k][_W] = MJ * (ETAx * fluxW_x + ETAy * fluxW_y + ETAz * fluxW_z);
  s_G[i][j][k][_E] = MJ * (ETAx * fluxE_x + ETAy * fluxE_y + ETAz * fluxE_z);

  s_H[i][j][k][_R] = MJ * (ZETAx * fluxR_x + ZETAy * fluxR_y + ZETAz * fluxR_z);
  s_H[i][j][k][_U] = MJ * (ZETAx * fluxU_x + ZETAy * fluxU_y + ZETAz * fluxU_z);
  s_H[i][j][k][_V] = MJ * (ZETAx * fluxV_x + ZETAy * fluxV_y + ZETAz * fluxV_z);
  s_H[i][j][k][_W] = MJ * (ZETAx * fluxW_x + ZETAy * fluxW_y + ZETAz * fluxW_z);
  s_H[i][j][k][_E] = MJ * (ZETAx * fluxE_x + ZETAy * fluxE_y + ZETAz * fluxE_z);

  __syncthreads();

  dfloat rhsU = 0, rhsV = 0, rhsW = 0, rhsR = 0, rhsE = 0;
  
  dfloat MJI = vgeo[gid +  _MJI*Np];
  
  // buoyancy term
  rhsW -= R * gravity;
  
  // loop of XI-grid lines
#pragma unroll Nq
  for(int n=0;n<Nq;++n){
    dfloat MJI_Dni = MJI * s_D[n][i];
    dfloat MJI_Dnj = MJI * s_D[n][j];
    dfloat MJI_Dnk = MJI * s_D[n][k];
    
    rhsR += MJI_Dni * s_F[n][ j][ k][ _R];
    rhsR += MJI_Dnj * s_G[i][ n][ k][ _R];
    rhsR += MJI_Dnk * s_H[i][ j][ n][ _R];
    
    rhsU += MJI_Dni * s_F[n][ j][ k][ _U];
    rhsU += MJI_Dnj * s_G[i][ n][ k][ _U];
    rhsU += MJI_Dnk * s_H[i][ j][ n][ _U];

    rhsV += MJI_Dni * s_F[n][ j][ k][ _V];
    rhsV += MJI_Dnj * s_G[i][ n][ k][ _V];
    rhsV += MJI_Dnk * s_H[i][ j][ n][ _V];

    rhsW += MJI_Dni * s_F[n][ j][ k][ _W];
    rhsW += MJI_Dnj * s_G[i][ n][ k][ _W];
    rhsW += MJI_Dnk * s_H[i][ j][ n][ _W];

    rhsE += MJI_Dni * s_F[n][ j][ k][ _E];
    rhsE += MJI_Dnj * s_G[i][ n][ k][ _E];
    rhsE += MJI_Dnk * s_H[i][ j][ n][ _E];
  }
  
  rhs[qid + Np*_U] += rhsU;
  rhs[qid + Np*_V] += rhsV;
  rhs[qid + Np*_W] += rhsW;
  rhs[qid + Np*_R] += rhsR;
  rhs[qid + Np*_E] += rhsE;

  // loop over moist variables
  // FIXME: Currently just passive advection
  //  TODO: This should probably be unrolled by some factor

  for(int m =0;m<nmoist;++m){
    int s = _nstate + m;
    
    __syncthreads();
    
    dfloat Qmoist = Q[qid + s*Np];
    
    dfloat fx = U * Rinv * Qmoist;
    dfloat fy = V * Rinv * Qmoist;
    dfloat fz = W * Rinv * Qmoist;
    
    s_F[i][j][k][1] = MJ * (XIx * fx + XIy * fy + XIz * fz);
    s_G[i][j][k][1] = MJ * (ETAx * fx + ETAy * fy + ETAz * fz);
    s_H[i][j][k][1] = MJ * (ZETAx * fx + ZETAy * fy + ZETAz * fz);
    
    __syncthreads();
    
    dfloat rhsmoist = 0;
#pragma unroll Nq
    for(int n=0;n<Nq;++n){
      dfloat MJI_Dni = MJI * s_D[n][i];
      dfloat MJI_Dnj = MJI * s_D[n][j];
      dfloat MJI_Dnk = MJI * s_D[n][k];
      
      rhsmoist += MJI_Dni * s_F[n][j][k][1];
      rhsmoist += MJI_Dnj * s_G[i][n][k][1];
      rhsmoist += MJI_Dnk * s_H[i][j][n][1];
    }
    rhs[qid + s*Np] += rhsmoist;
  }

  // Loop over trace variables
  // TODO: This should probably be unrolled by some factor
  dfloat rhstrace = 0;
  for(int m=0;m<ntrace;++m){

    int s = _nstate + nmoist + m;
    
    __syncthreads();
    
    dfloat Qtrace = Q[qid+s*Np];
    
    dfloat fx = U * Rinv * Qtrace;
    dfloat fy = V * Rinv * Qtrace;
    dfloat fz = W * Rinv * Qtrace;

    s_F[i][j][k][1] = MJ * (XIx * fx + XIy * fy + XIz * fz);
    s_G[i][j][k][1] = MJ * (ETAx * fx + ETAy * fy + ETAz * fz);
    s_H[i][j][k][1] = MJ * (ZETAx * fx + ZETAy * fy + ZETAz * fz);

    __syncthreads();

    // TODO: Prefetch MJI and rhs
    rhstrace = 0;
#pragma unroll Nq
    for(int n=0;n<Nq;++n){
      dfloat MJI_Dni = MJI * s_D[n][i];
      dfloat MJI_Dnj = MJI * s_D[n][j];
      dfloat MJI_Dnk = MJI * s_D[n][k];
      
      rhstrace += MJI_Dni * s_F[n][j][k][1];
      rhstrace += MJI_Dnj * s_G[i][n][k][1];
      rhstrace += MJI_Dnk * s_H[i][j][n][1];
    }
    
    rhs[qid+s*Np] += rhstrace;
  }
}

template <int Nq, int Np,  int nmoist, int ntrace, int nvar>
  __global__ void volumerhs_v3(dfloat * __restrict__ rhs,
			       const dfloat * __restrict__ Q,
			       const dfloat * __restrict__ vgeo,
			       const dfloat gravity,
			       const dfloat * __restrict__ D,
			       const int nelem){

  __shared__ dfloat s_D[Nq][Nq];
  __shared__ dfloat s_F[Nq][Nq][_nstate];
  __shared__ dfloat s_G[Nq][Nq][_nstate];

  dfloat r_rhsR[Nq];
  dfloat r_rhsU[Nq];
  dfloat r_rhsV[Nq];
  dfloat r_rhsW[Nq];
  dfloat r_rhsE[Nq];  

  int e = blockIdx.x;
  int j = threadIdx.y;
  int i = threadIdx.x;
  
  s_D[j][i] = D[j*Nq+i];
  
#pragma unroll Nq  
  for(int k=0;k<Nq;++k){
    r_rhsR[k] = 0;
    r_rhsU[k] = 0;
    r_rhsV[k] = 0;
    r_rhsW[k] = 0;
    r_rhsE[k] = 0;
  }

#pragma unroll Nq
  for(int k=0;k<Nq;++k){

    __syncthreads();

    // Load values will need into registers
    int gid = i + j*Nq + k*Nq*Nq + e*Np*_nvgeo;
    
    dfloat MJ = vgeo[gid + _MJ*Np];
    dfloat XIx = vgeo[gid + _XIx*Np];
    dfloat XIy = vgeo[gid + _XIy*Np];
    dfloat XIz = vgeo[gid + _XIz*Np];
    dfloat ETAx = vgeo[gid + _ETAx*Np];
    dfloat ETAy = vgeo[gid + _ETAy*Np];
    dfloat ETAz = vgeo[gid + _ETAz*Np];
    dfloat ZETAx = vgeo[gid + _ZETAx*Np];
    dfloat ZETAy = vgeo[gid + _ZETAy*Np];
    dfloat ZETAz = vgeo[gid + _ZETAz*Np];
    dfloat z = vgeo[gid +  _z*Np];
    
    
    int qid = i + j*Nq + k*Nq*Nq + e*Np*nvar;

    dfloat R = Q[qid + _R*Np];
    dfloat U = Q[qid + _U*Np];
    dfloat V = Q[qid + _V*Np];
    dfloat W = Q[qid + _W*Np];
    dfloat E = Q[qid + _E*Np];
    
    dfloat P = gdm1*(E - (U*U + V*V + W*W)/(2*R) - R*gravity*z);
    
    dfloat Rinv = 1 / R;  

    dfloat fluxR_x = U;
    dfloat fluxU_x = Rinv * U * U + P;
    dfloat fluxV_x = Rinv * U * V;
    dfloat fluxW_x = Rinv * U * W;
    dfloat fluxE_x = Rinv * U * (E + P);
    
    dfloat fluxR_y = V;
    dfloat fluxU_y = Rinv * V * U;
    dfloat fluxV_y = Rinv * V * V + P;
    dfloat fluxW_y = Rinv * V * W;
    dfloat fluxE_y = Rinv * V * (E + P);
    
    dfloat fluxR_z = W;
    dfloat fluxU_z = Rinv * W * U;
    dfloat fluxV_z = Rinv * W * V;
    dfloat fluxW_z = Rinv * W * W + P;
    dfloat fluxE_z = Rinv * W * (E + P);
    
    s_F[i][j][ _R] = MJ * (XIx * fluxR_x + XIy * fluxR_y + XIz * fluxR_z);
    s_F[i][j][ _U] = MJ * (XIx * fluxU_x + XIy * fluxU_y + XIz * fluxU_z);
    s_F[i][j][ _V] = MJ * (XIx * fluxV_x + XIy * fluxV_y + XIz * fluxV_z);
    s_F[i][j][ _W] = MJ * (XIx * fluxW_x + XIy * fluxW_y + XIz * fluxW_z);
    s_F[i][j][ _E] = MJ * (XIx * fluxE_x + XIy * fluxE_y + XIz * fluxE_z);
    
    s_G[i][j][ _R] = MJ * (ETAx * fluxR_x + ETAy * fluxR_y + ETAz * fluxR_z);
    s_G[i][j][ _U] = MJ * (ETAx * fluxU_x + ETAy * fluxU_y + ETAz * fluxU_z);
    s_G[i][j][ _V] = MJ * (ETAx * fluxV_x + ETAy * fluxV_y + ETAz * fluxV_z);
    s_G[i][j][ _W] = MJ * (ETAx * fluxW_x + ETAy * fluxW_y + ETAz * fluxW_z);
    s_G[i][j][ _E] = MJ * (ETAx * fluxE_x + ETAy * fluxE_y + ETAz * fluxE_z);
    
    dfloat r_HR = MJ * (ZETAx * fluxR_x + ZETAy * fluxR_y + ZETAz * fluxR_z);
    dfloat r_HU = MJ * (ZETAx * fluxU_x + ZETAy * fluxU_y + ZETAz * fluxU_z);
    dfloat r_HV = MJ * (ZETAx * fluxV_x + ZETAy * fluxV_y + ZETAz * fluxV_z);
    dfloat r_HW = MJ * (ZETAx * fluxW_x + ZETAy * fluxW_y + ZETAz * fluxW_z);
    dfloat r_HE = MJ * (ZETAx * fluxE_x + ZETAy * fluxE_y + ZETAz * fluxE_z);
    
    // one shared access per 10 flops
#pragma unroll Nq
    for(int n=0;n<Nq;++n){
      dfloat  Dkn = s_D[k][n];
      
      r_rhsR[n] += Dkn * r_HR;
      r_rhsU[n] += Dkn * r_HU;
      r_rhsV[n] += Dkn * r_HV;
      r_rhsW[n] += Dkn * r_HW;
      r_rhsE[n] += Dkn * r_HE;
    }

    r_rhsW[k] -= MJ * R * gravity;

    __syncthreads();

    // loop of XI-grid lines
#pragma unroll Nq
    for(int n=0;n<Nq;++n){
      dfloat Dni = s_D[n][i];
      dfloat Dnj = s_D[n][j];
      
      r_rhsR[k] += Dni * s_F[n][j][_R];
      r_rhsR[k] += Dnj * s_G[i][n][_R];

      r_rhsU[k] += Dni * s_F[n][j][_U];
      r_rhsU[k] += Dnj * s_G[i][n][_U];

      r_rhsV[k] += Dni * s_F[n][j][_V];
      r_rhsV[k] += Dnj * s_G[i][n][_V];

      r_rhsW[k] += Dni * s_F[n][j][_W];
      r_rhsW[k] += Dnj * s_G[i][n][_W];

      r_rhsE[k] += Dni * s_F[n][j][_E];
      r_rhsE[k] += Dnj * s_G[i][n][_E];
    }
  }
  
#pragma unroll Nq
  for(int k=0;k<Nq;++k){
    int gid = i + j*Nq + k*Nq*Nq + e*Np*_nvgeo;
    dfloat MJI = vgeo[gid +  _MJI*Np];
    
    int qid = i + j*Nq + k*Nq*Nq + e*Np*nvar;
    
    rhs[qid+_U*Np] += MJI*r_rhsU[k];
    rhs[qid+_V*Np] += MJI*r_rhsV[k];
    rhs[qid+_W*Np] += MJI*r_rhsW[k];
    rhs[qid+_R*Np] += MJI*r_rhsR[k];
    rhs[qid+_E*Np] += MJI*r_rhsE[k];
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

void shiftEntries(int start, int end, dfloat shift, dfloat *q, dfloat *c_q){

  for(int n=start;n<end;++n){
    q[n] += shift;
  }

  cudaMemcpy(c_q+start, q+start, (end-start)*sizeof(dfloat), cudaMemcpyHostToDevice);
  
}
		      
int main(int argc, char **argv){

  srand48(1234);

  const int N = POLYNOMIAL_ORDER;
  const int nelem = atoi(argv[1]);
  
  const int nmoist = 0;
  const int ntrace = 0;
  
  const int nvar = _nstate + nmoist + ntrace;

  const int Nq = N+1;
  const int Np = Nq*Nq*Nq;
  
  const int Ntotal = Np*nelem*nvar;

  dfloat *Q, *c_Q;
  randArray(Ntotal, 0., 1., &Q, &c_Q);

  for(int e=0;e<nelem;++e){
    for(int n=0;n<Np;++n){
      int idR = n + _R*Np + e*nvar*Np;
      int idE = n + _E*Np + e*nvar*Np;

      Q[idR] += 2.;
      Q[idE] += 20.;
      
    }
  }

  cudaMemcpy(c_Q, Q, nelem*nvar*Np*sizeof(dfloat), cudaMemcpyHostToDevice);
  
  const int Gtotal = Np*nelem*_nvgeo;

  dfloat *vgeo, *c_vgeo;
  randArray(Gtotal, 0, 1., &vgeo, &c_vgeo);

  // Make sure the entries of the mass matrix satisfy the inverse relation
  for(int e=0;e<nelem;++e){
    for(int n=0;n<Np;++n){
      int idMJ = n + _MJ*Np + e*_nvgeo*Np;
      int idMJI = n + _MJI*Np + e*_nvgeo*Np;

      vgeo[idMJ] += 3;
      vgeo[idMJI] = 1./vgeo[idMJ];
      
    }
  }
  cudaMemcpy(c_vgeo, vgeo, nelem*_nvgeo*Np*sizeof(dfloat), cudaMemcpyHostToDevice);
  
  dfloat *D, *c_D;
  randArray(Nq*Nq, 1., 1., &D, &c_D);

  dfloat *rhs_v2, *c_rhs_v2;
  dfloat *rhs_v3, *c_rhs_v3;

  srand48(1234);
  randArray(Ntotal, 1., 1., &rhs_v2, &c_rhs_v2);

  srand48(1234);
  randArray(Ntotal, 1., 1., &rhs_v3, &c_rhs_v3);

  dim3 G(nelem,1,1);
  dim3 B2(Nq,Nq,Nq);
  dim3 B3(Nq,Nq,1);

  int Ntests = 1;
  for(int test=0;test<Ntests;++test){
    volumerhs_v2<Nq, Np, nmoist, ntrace, nvar> <<< G, B2 >>> (c_rhs_v2, c_Q, c_vgeo, grav, c_D, nelem);
  }
  for(int test=0;test<Ntests;++test){
    volumerhs_v3<Nq, Np, nmoist, ntrace, nvar> <<< G, B3 >>> (c_rhs_v3, c_Q, c_vgeo, grav, c_D, nelem);
  }

  cudaMemcpy(rhs_v2, c_rhs_v2, Ntotal*sizeof(dfloat), cudaMemcpyDeviceToHost);
  cudaMemcpy(rhs_v3, c_rhs_v3, Ntotal*sizeof(dfloat), cudaMemcpyDeviceToHost);

  dfloat maxDiff = 0;
  for(int e=0;e<nelem;++e){
    for(int v=0;v<nvar;++v){
      for(int n=0;n<Np;++n){
	int id = n + v*Np + e*nvar*Np;
	
	dfloat diff = fabs(rhs_v2[id]-rhs_v3[id]);

	if(diff>maxDiff)
	  maxDiff = diff;

	//	printf("id: %d, rhs:%lf\n", id, rhs_v3[id]);
	//printf("(e: %d, v: %d, n: %d) v2 %lg, v3 %lg, diff %lg, maxdiff = %lg\n", e, v, n, rhs_v2[id], rhs_v3[id], diff, maxDiff);

      }
    }
  }
  printf("max diff = %lg\n", maxDiff);
  
  cudaDeviceSynchronize();

  exit(0);
  return 0;
}
