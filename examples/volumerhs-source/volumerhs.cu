#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <cuda_profiler_api.h>

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

// note the order of the fields below is also assumed in the code.
const int64_t _nstate = 5;

const int64_t _R = 0, _U  = 1, _V  = 2, _W = 3, _E = 4;

const int64_t _nvgeo = 14;
const int64_t _XIx   = 0;
const int64_t _ETAx  = 1;
const int64_t _ZETAx = 2;
const int64_t _XIy   = 3;
const int64_t _ETAy  = 4;
const int64_t _ZETAy = 5;
const int64_t _XIz   = 6;
const int64_t _ETAz  = 7;
const int64_t _ZETAz = 8;
const int64_t _MJ    = 9;
const int64_t _MJI   = 10;
const int64_t _x     = 11;
const int64_t _y     = 12;
const int64_t _z     = 13;

#define gdm1  ((dfloat) 0.4)
#define pi    ((dfloat) 3.14)


template <int64_t Nq, int64_t Np, int64_t nvar>
  __global__ void volumerhs(dfloat * __restrict__ rhs,
                            const dfloat * __restrict__ Q,
                            const dfloat * __restrict__ vgeo,
                            const dfloat * __restrict__ D,
                            const int64_t nelem,
                            const dfloat t){

    __shared__ dfloat s_D[Nq][Nq];
    __shared__ dfloat s_F[Nq][Nq][Nq][_nstate];
    __shared__ dfloat s_G[Nq][Nq][Nq][_nstate];
    __shared__ dfloat s_H[Nq][Nq][Nq][_nstate];

    int64_t e = blockIdx.x;
    int64_t k = threadIdx.z;
    int64_t j = threadIdx.y;
    int64_t i = threadIdx.x;

    if(k==0) s_D[j][i] = D[j*Nq+i];

    // Load values will need int64_to registers
    int64_t gid = i + j*Nq + k*Nq*Nq + e*Np*_nvgeo;

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
    dfloat x = vgeo[gid +  _x*Np];
    dfloat y = vgeo[gid +  _y*Np];
    dfloat z = vgeo[gid +  _z*Np];


    int64_t qid = i + j*Nq + k*Nq*Nq + e*Np*nvar;

    dfloat R = Q[qid + _R*Np];
    dfloat U = Q[qid + _U*Np];
    dfloat V = Q[qid + _V*Np];
    dfloat W = Q[qid + _W*Np];
    dfloat E = Q[qid + _E*Np];

    dfloat P = gdm1*(E - (U*U + V*V + W*W)/(2*R));

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

    s_H[i][j][k][_R]  = MJ * (ZETAx * fluxR_x + ZETAy * fluxR_y + ZETAz * fluxR_z);
    s_H[i][j][k][_U]  = MJ * (ZETAx * fluxU_x + ZETAy * fluxU_y + ZETAz * fluxU_z);
    s_H[i][j][k][_V]  = MJ * (ZETAx * fluxV_x + ZETAy * fluxV_y + ZETAz * fluxV_z);
    s_H[i][j][k][_W]  = MJ * (ZETAx * fluxW_x + ZETAy * fluxW_y + ZETAz * fluxW_z);
    s_H[i][j][k][_E]  = MJ * (ZETAx * fluxE_x + ZETAy * fluxE_y + ZETAz * fluxE_z);

    dfloat r_rhsR = pi*(-sin(pi*t)*sin(pi*x)*cos(pi*y)*cos(pi*z) -
                        2*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y) +
                        sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        3*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*z) +
                        2*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        3*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        3*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z));
    dfloat r_rhsU = pi*(-10*sin(pi*t)*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        15*sin(pi*t)*sin(pi*x)*cos(pi*z) -
                        15*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        10*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        5*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        30*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*z) -
                        3*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) -
                        15*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y) +
                        9*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        15*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        6*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*x)*cos(pi*y) +
                        18*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        2*cos(pi*t)*cos(pi*x)*cos(pi*z))*cos(pi*y)/5;
    dfloat r_rhsV = pi*(-10*sin(pi*t)*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        15*sin(pi*t)*cos(pi*y)*cos(pi*z) +
                        3*sin(pi*x)*sin(pi*y)*sin(pi*z)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        9*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        10*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        5*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        6*sin(pi*x)*sin(pi*y)*sin(pi*z)*cos(pi*t)*cos(pi*y) -
                        18*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        15*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y) +
                        15*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        15*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        2*sin(pi*y)*cos(pi*t)*cos(pi*z) +
                        30*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z))*sin(pi*x)/5;
    dfloat r_rhsW = pi*(-10*sin(pi*t)*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        15*sin(pi*t) -
                        15*sin(pi*x)*sin(pi*y)*sin(pi*(x + y))*cos(pi*t)*cos(pi*z) +
                        15*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*z) +
                        36*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        18*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        4*cos(pi*t)*cos(pi*x)*cos(pi*y) +
                        18*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        4*cos(pi*t)*cos(pi*y) +
                        30*cos(pi*t)*cos(pi*z)*cos(pi*(x + y)) -
                        2*cos(pi*t))*sin(pi*x)*sin(pi*z)*cos(pi*y)/5;
    dfloat r_rhsE = pi*(3*(1 - cos(2*pi*x))*
                        (1 - cos(4*pi*z))*(cos(2*pi*t) + 1)*
                        (cos(2*pi*y) + 1)/512 - 5*sin(pi*t)*
                        sin(pi*x)*cos(pi*y)*cos(pi*z) +
                        4*sin(pi*x)*sin(pi*y)*sin(pi*z)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        8*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y) -
                        2*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        9*sin(pi*x)*sin(pi*y)*sin(pi*z)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        18*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        4*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        3*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        8*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) -
                        6*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        14*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        9*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) -
                        7*sin(pi*x)*sin(pi*z)*cos(pi*t)*cos(pi*y) -
                        18*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        7*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) -
                        700*sin(pi*x)*sin(pi*y)*cos(pi*t)*cos(pi*z) +
                        14*sin(pi*x)*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z) +
                        700*sin(pi*x)*cos(pi*t)*cos(pi*y)*cos(pi*z) +
                        700*cos(pi*t)*cos(pi*x)*cos(pi*y)*cos(pi*z))/5;

    __syncthreads();

    // loop of XI-grid lines
#pragma unroll Nq
    for(int64_t n=0;n<Nq;++n){
      dfloat Dni = s_D[n][i];
      dfloat Dnj = s_D[n][j];
      dfloat Dnk = s_D[n][k];

      r_rhsR += Dni * s_F[n][j][k][_R];
      r_rhsR += Dnj * s_G[i][n][k][_R];
      r_rhsR += Dnk * s_H[i][j][n][_R];

      r_rhsU += Dni * s_F[n][j][k][_U];
      r_rhsU += Dnj * s_G[i][n][k][_U];
      r_rhsU += Dnk * s_H[i][j][n][_U];

      r_rhsV += Dni * s_F[n][j][k][_V];
      r_rhsV += Dnj * s_G[i][n][k][_V];
      r_rhsV += Dnk * s_H[i][j][n][_V];

      r_rhsW += Dni * s_F[n][j][k][_W];
      r_rhsW += Dnj * s_G[i][n][k][_W];
      r_rhsW += Dnk * s_H[i][j][n][_W];

      r_rhsE += Dni * s_F[n][j][k][_E];
      r_rhsE += Dnj * s_G[i][n][k][_E];
      r_rhsE += Dnk * s_H[i][j][n][_E];
    }

    dfloat MJI = vgeo[gid +  _MJI*Np];

    rhs[qid+_U*Np] += MJI*r_rhsU;
    rhs[qid+_V*Np] += MJI*r_rhsV;
    rhs[qid+_W*Np] += MJI*r_rhsW;
    rhs[qid+_R*Np] += MJI*r_rhsR;
    rhs[qid+_E*Np] += MJI*r_rhsE;
  }

template <int64_t Nq, int64_t Np, int64_t nvar>
  __global__ void volumerhs_trigpi(dfloat * __restrict__ rhs,
                                   const dfloat * __restrict__ Q,
                                   const dfloat * __restrict__ vgeo,
                                   const dfloat * __restrict__ D,
                                   const int64_t nelem,
                                   const dfloat t){

    __shared__ dfloat s_D[Nq][Nq];
    __shared__ dfloat s_F[Nq][Nq][Nq][_nstate];
    __shared__ dfloat s_G[Nq][Nq][Nq][_nstate];
    __shared__ dfloat s_H[Nq][Nq][Nq][_nstate];

    int64_t e = blockIdx.x;
    int64_t k = threadIdx.z;
    int64_t j = threadIdx.y;
    int64_t i = threadIdx.x;

    if(k==0) s_D[j][i] = D[j*Nq+i];

    // Load values will need int64_to registers
    int64_t gid = i + j*Nq + k*Nq*Nq + e*Np*_nvgeo;

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
    dfloat x = vgeo[gid +  _x*Np];
    dfloat y = vgeo[gid +  _y*Np];
    dfloat z = vgeo[gid +  _z*Np];


    int64_t qid = i + j*Nq + k*Nq*Nq + e*Np*nvar;

    dfloat R = Q[qid + _R*Np];
    dfloat U = Q[qid + _U*Np];
    dfloat V = Q[qid + _V*Np];
    dfloat W = Q[qid + _W*Np];
    dfloat E = Q[qid + _E*Np];

    dfloat P = gdm1*(E - (U*U + V*V + W*W)/(2*R));

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

    s_H[i][j][k][_R]  = MJ * (ZETAx * fluxR_x + ZETAy * fluxR_y + ZETAz * fluxR_z);
    s_H[i][j][k][_U]  = MJ * (ZETAx * fluxU_x + ZETAy * fluxU_y + ZETAz * fluxU_z);
    s_H[i][j][k][_V]  = MJ * (ZETAx * fluxV_x + ZETAy * fluxV_y + ZETAz * fluxV_z);
    s_H[i][j][k][_W]  = MJ * (ZETAx * fluxW_x + ZETAy * fluxW_y + ZETAz * fluxW_z);
    s_H[i][j][k][_E]  = MJ * (ZETAx * fluxE_x + ZETAy * fluxE_y + ZETAz * fluxE_z);

    dfloat r_rhsR = pi*(-sinpi(t)*sinpi(x)*cospi(y)*cospi(z) -
                        2*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) -
                        sinpi(x)*sinpi(z)*cospi(t)*cospi(y) +
                        sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        3*sinpi(x)*sinpi(y)*cospi(t)*cospi(z) +
                        2*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        3*sinpi(x)*cospi(t)*cospi(y)*cospi(z) +
                        3*cospi(t)*cospi(x)*cospi(y)*cospi(z));
    dfloat r_rhsU = pi*(-10*sinpi(t)*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        15*sinpi(t)*sinpi(x)*cospi(z) -
                        15*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) -
                        10*sinpi(x)*sinpi(z)*cospi(t)*cospi(y)*cospi(z) +
                        5*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        30*sinpi(x)*sinpi(y)*cospi(t)*cospi(z) -
                        3*sinpi(x)*sinpi(z)*cospi(t)*cospi(x)*cospi(y)*cospi(z) -
                        15*sinpi(x)*sinpi(z)*cospi(t)*cospi(y) +
                        9*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        15*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        6*sinpi(x)*sinpi(z)*cospi(t)*cospi(x)*cospi(y) +
                        18*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        2*cospi(t)*cospi(x)*cospi(z))*cospi(y)/5;
    dfloat r_rhsV = pi*(-10*sinpi(t)*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        15*sinpi(t)*cospi(y)*cospi(z) +
                        3*sinpi(x)*sinpi(y)*sinpi(z)*cospi(t)*cospi(y)*cospi(z) -
                        9*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) -
                        10*sinpi(x)*sinpi(z)*cospi(t)*cospi(y)*cospi(z) +
                        5*sinpi(x)*cospi(t)*cospi(y)*cospi(z) +
                        6*sinpi(x)*sinpi(y)*sinpi(z)*cospi(t)*cospi(y) -
                        18*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) -
                        15*sinpi(x)*sinpi(z)*cospi(t)*cospi(y) +
                        15*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        15*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        2*sinpi(y)*cospi(t)*cospi(z) +
                        30*cospi(t)*cospi(x)*cospi(y)*cospi(z))*sinpi(x)/5;
    dfloat r_rhsW = pi*(-10*sinpi(t)*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        15*sinpi(t) -
                        15*sinpi(x)*sinpi(y)*sinpi((x + y))*cospi(t)*cospi(z) +
                        15*sinpi(x)*cospi(t)*cospi(x)*cospi(z) +
                        36*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        18*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        4*cospi(t)*cospi(x)*cospi(y) +
                        18*cospi(t)*cospi(y)*cospi(z) -
                        4*cospi(t)*cospi(y) +
                        30*cospi(t)*cospi(z)*cospi((x + y)) -
                        2*cospi(t))*sinpi(x)*sinpi(z)*cospi(y)/5;
    dfloat r_rhsE = pi*(3*(1 - cos(2*pi*x))*
                        (1 - cos(4*pi*z))*(cos(2*pi*t) + 1)*
                        (cos(2*pi*y) + 1)/512 - 5*sinpi(t)*
                        sinpi(x)*cospi(y)*cospi(z) +
                        4*sinpi(x)*sinpi(y)*sinpi(z)*cospi(t)*cospi(y)*cospi(z) +
                        8*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) +
                        sinpi(x)*sinpi(z)*cospi(t)*cospi(y) -
                        2*sinpi(x)*cospi(t)*cospi(y)*cospi(z) +
                        9*sinpi(x)*sinpi(y)*sinpi(z)*cospi(t)*cospi(y)*cospi(z) +
                        18*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) -
                        4*sinpi(x)*sinpi(z)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        3*sinpi(x)*sinpi(z)*cospi(t)*cospi(y)*cospi(z) -
                        8*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) -
                        6*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        14*sinpi(x)*sinpi(y)*cospi(t)*cospi(y)*cospi(z) -
                        9*sinpi(x)*sinpi(z)*cospi(t)*cospi(x)*cospi(y)*cospi(z) -
                        7*sinpi(x)*sinpi(z)*cospi(t)*cospi(y) -
                        18*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        7*sinpi(x)*cospi(t)*cospi(y)*cospi(z) -
                        700*sinpi(x)*sinpi(y)*cospi(t)*cospi(z) +
                        14*sinpi(x)*cospi(t)*cospi(x)*cospi(y)*cospi(z) +
                        700*sinpi(x)*cospi(t)*cospi(y)*cospi(z) +
                        700*cospi(t)*cospi(x)*cospi(y)*cospi(z))/5;

    __syncthreads();

    // loop of XI-grid lines
#pragma unroll Nq
    for(int64_t n=0;n<Nq;++n){
      dfloat Dni = s_D[n][i];
      dfloat Dnj = s_D[n][j];
      dfloat Dnk = s_D[n][k];

      r_rhsR += Dni * s_F[n][j][k][_R];
      r_rhsR += Dnj * s_G[i][n][k][_R];
      r_rhsR += Dnk * s_H[i][j][n][_R];

      r_rhsU += Dni * s_F[n][j][k][_U];
      r_rhsU += Dnj * s_G[i][n][k][_U];
      r_rhsU += Dnk * s_H[i][j][n][_U];

      r_rhsV += Dni * s_F[n][j][k][_V];
      r_rhsV += Dnj * s_G[i][n][k][_V];
      r_rhsV += Dnk * s_H[i][j][n][_V];

      r_rhsW += Dni * s_F[n][j][k][_W];
      r_rhsW += Dnj * s_G[i][n][k][_W];
      r_rhsW += Dnk * s_H[i][j][n][_W];

      r_rhsE += Dni * s_F[n][j][k][_E];
      r_rhsE += Dnj * s_G[i][n][k][_E];
      r_rhsE += Dnk * s_H[i][j][n][_E];
    }

    dfloat MJI = vgeo[gid +  _MJI*Np];

    rhs[qid+_U*Np] += MJI*r_rhsU;
    rhs[qid+_V*Np] += MJI*r_rhsV;
    rhs[qid+_W*Np] += MJI*r_rhsW;
    rhs[qid+_R*Np] += MJI*r_rhsR;
    rhs[qid+_E*Np] += MJI*r_rhsE;
  }

void randArray(int64_t N, dfloat base, dfloat range, dfloat **q, dfloat **c_q){

  *q = (dfloat*) calloc(N, sizeof(dfloat));
  cudaMalloc(c_q, N*sizeof(dfloat));

  for(int64_t n=0;n<N;++n){
    q[0][n] = base + drand48()*range;
  }

  cudaMemcpy(c_q[0], q[0], N*sizeof(dfloat), cudaMemcpyHostToDevice);

}

int main(int argc, char **argv){

  srand48(1234);

  const int64_t N = POLYNOMIAL_ORDER;
  const int64_t nelem = 4000;

  const int64_t Nq = N+1;
  const int64_t Np = Nq*Nq*Nq;

  const int64_t Ntotal = Np*nelem*_nstate;

  const dfloat t = 1;

  dfloat *Q, *c_Q;
  randArray(Ntotal, 0., 1., &Q, &c_Q);

  for(int64_t e=0;e<nelem;++e){
    for(int64_t n=0;n<Np;++n){
      int64_t idR = n + _R*Np + e*_nstate*Np;
      int64_t idE = n + _E*Np + e*_nstate*Np;

      Q[idR] += 2.;
      Q[idE] += 20.;

    }
  }

  cudaMemcpy(c_Q, Q, nelem*_nstate*Np*sizeof(dfloat), cudaMemcpyHostToDevice);

  const int64_t Gtotal = Np*nelem*_nvgeo;

  dfloat *vgeo, *c_vgeo;
  randArray(Gtotal, 0, 1., &vgeo, &c_vgeo);

  // Make sure the entries of the mass matrix satisfy the inverse relation
  for(int64_t e=0;e<nelem;++e){
    for(int64_t n=0;n<Np;++n){
      int64_t idMJ = n + _MJ*Np + e*_nvgeo*Np;
      int64_t idMJI = n + _MJI*Np + e*_nvgeo*Np;

      vgeo[idMJ] += 3;
      vgeo[idMJI] = 1./vgeo[idMJ];

    }
  }
  cudaMemcpy(c_vgeo, vgeo, nelem*_nvgeo*Np*sizeof(dfloat), cudaMemcpyHostToDevice);

  dfloat *D, *c_D;
  randArray(Nq*Nq, 1., 1., &D, &c_D);

  dfloat *rhs, *c_rhs;

  srand48(1234);
  randArray(Ntotal, 1., 1., &rhs, &c_rhs);

  dim3 G(nelem,1,1);
  dim3 B2(Nq,Nq,Nq);
  dim3 B3(Nq,Nq,1);

  cudaProfilerStart();
  volumerhs<Nq, Np, _nstate> <<< G, B2 >>> (c_rhs, c_Q, c_vgeo, c_D, nelem, t);
  volumerhs_trigpi<Nq, Np, _nstate> <<< G, B2 >>> (c_rhs, c_Q, c_vgeo, c_D, nelem, t);
  cudaProfilerStop();

  cudaDeviceSynchronize();

  exit(0);
  return 0;
}
