using Random
using CUDAdrv
using CUDAnative
using CuArrays
using StaticArrays
using GPUifyLoops # for @unroll macro

# Base.@propagate_inbounds function Base.getindex(v::MArray, i::Int)
#     @boundscheck checkbounds(v,i)
#     T = eltype(v)
#
#     @assert isbitstype(T) "Only isbits is supported on the device"
#     return GC.@preserve v begin
#         ptr  = pointer_from_objref(v)
#         dptr = reinterpret(CUDAnative.DevicePtr{T, CUDAnative.AS.Local}, ptr)
#         unsafe_load(dptr, i)
#     end
# end
#
# Base.@propagate_inbounds function Base.setindex!(v::MArray, val, i::Int)
#     @boundscheck checkbounds(v,i)
#     T = eltype(v)
#
#     @assert isbitstype(T) "Only isbits is supported on the device"
#     GC.@preserve v begin
#     ptr  = pointer_from_objref(v)
#         dptr = reinterpret(CUDAnative.DevicePtr{T, CUDAnative.AS.Local}, ptr)
#         unsafe_store!(dptr, convert(T, val), i)
#     end
#
#     return val
# end

for (f, T) in Base.Iterators.product((:add, :mul, :sub), (Float32, Float64))
    name = Symbol("$(f)_float_contract")
    if T === Float32
        llvmt = "float"
    elseif T === Float64
        llvmt = "double"
    end

    ir = """
    %x = f$f contract $llvmt %0, %1
    ret $llvmt %x
    """
    @eval begin
        # the @pure is necessary so that we can constant propagate.
        Base.@pure function $name(a::$T, b::$T)
            @Base._inline_meta
            Base.llvmcall($ir, $T, Tuple{$T, $T}, a, b)
        end
    end
end

# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo

Base.@irrational gdm1 0.4 BigFloat(0.4)

function volumerhs!(::Val{N}, rhs, Q, vgeo, D, nelem, t) where {N}
  Nq = N + 1

  s_D = @cuStaticSharedMem eltype(D) (Nq, Nq)
  s_F = @cuStaticSharedMem eltype(Q) (Nq, Nq, Nq, _nstate)
  s_G = @cuStaticSharedMem eltype(Q) (Nq, Nq, Nq, _nstate)
  s_H = @cuStaticSharedMem eltype(Q) (Nq, Nq, Nq, _nstate)

  e = blockIdx().x
  k = threadIdx().z
  j = threadIdx().y
  i = threadIdx().x

  @inbounds begin
    # fetch D into shared
    k == 1 && (s_D[i, j] = D[i, j])

    # Load values will need into registers
    MJ = vgeo[i, j, k, _MJ, e]
    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
    x, y, z = vgeo[i,j,k,_x,e], vgeo[i,j,k,_y,e], vgeo[i,j,k,_z,e]

    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ))

    ρinv = 1 / ρ

    fluxρ_x = U
    fluxU_x = ρinv * U * U + P
    fluxV_x = ρinv * U * V
    fluxW_x = ρinv * U * W
    fluxE_x = ρinv * U * (E + P)

    fluxρ_y = V
    fluxU_y = ρinv * V * U
    fluxV_y = ρinv * V * V + P
    fluxW_y = ρinv * V * W
    fluxE_y = ρinv * V * (E + P)

    fluxρ_z = W
    fluxU_z = ρinv * W * U
    fluxV_z = ρinv * W * V
    fluxW_z = ρinv * W * W + P
    fluxE_z = ρinv * W * (E + P)

    s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
    s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
    s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
    s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
    s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

    s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
    s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
    s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
    s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
    s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

    s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
    s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
    s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
    s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
    s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

    r_rhsρ = pi*(-CUDAnative.sin(pi*t)*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 2*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) +
                 CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 3*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*z) +
                 2*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 3*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 3*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z))
    r_rhsU = pi*(-10*CUDAnative.sin(pi*t)*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 15*CUDAnative.sin(pi*t)*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*z) -
                 15*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 10*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 5*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 30*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*z) -
                 3*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 15*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) +
                 9*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 15*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 6*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y) +
                 18*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 2*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*z))*CUDAnative.cos(pi*y)/5
    r_rhsV = pi*(-10*CUDAnative.sin(pi*t)*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 15*CUDAnative.sin(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 3*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 9*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 10*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 5*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 6*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) -
                 18*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 15*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) +
                 15*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 15*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 2*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*z) +
                 30*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z))*CUDAnative.sin(pi*x)/5
    r_rhsW = pi*(-10*CUDAnative.sin(pi*t)*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 15*CUDAnative.sin(pi*t) -
                 15*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.sin(pi*(x + y))*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*z) +
                 15*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*z) +
                 36*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 18*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 4*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y) +
                 18*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 4*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) +
                 30*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*z)*CUDAnative.cos(pi*(x + y)) -
                 2*CUDAnative.cos(pi*t))*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*y)/5
    r_rhsE = pi*(3*(1 - CUDAnative.cos(2*pi*x))*
                 (1 - CUDAnative.cos(4*pi*z))*(CUDAnative.cos(2*pi*t) + 1)*
                 (CUDAnative.cos(2*pi*y) + 1)/512 - 5*CUDAnative.sin(pi*t)*
                 CUDAnative.sin(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 4*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 8*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) -
                 2*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 9*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 18*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 4*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 3*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 8*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 6*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 14*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 9*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 7*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*z)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y) -
                 18*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 7*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) -
                 700*CUDAnative.sin(pi*x)*CUDAnative.sin(pi*y)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*z) +
                 14*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 700*CUDAnative.sin(pi*x)*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z) +
                 700*CUDAnative.cos(pi*t)*CUDAnative.cos(pi*x)*CUDAnative.cos(pi*y)*CUDAnative.cos(pi*z))/5

    sync_threads()

    # loop of ξ-grid lines
    @unroll for n = 1:Nq
      Dni = s_D[n, i]
      Dnj = s_D[n, j]
      Dnk = s_D[n, k]

      r_rhsρ += Dni * s_F[n, j, k, _ρ]
      r_rhsρ += Dnj * s_G[i, n, k, _ρ]
      r_rhsρ += Dnk * s_H[i, j, n, _ρ]

      r_rhsU += Dni * s_F[n, j, k, _U]
      r_rhsU += Dnj * s_G[i, n, k, _U]
      r_rhsU += Dnk * s_H[i, j, n, _U]

      r_rhsV += Dni * s_F[n, j, k, _V]
      r_rhsV += Dnj * s_G[i, n, k, _V]
      r_rhsV += Dnk * s_H[i, j, n, _V]

      r_rhsW += Dni * s_F[n, j, k, _W]
      r_rhsW += Dnj * s_G[i, n, k, _W]
      r_rhsW += Dnk * s_H[i, j, n, _W]

      r_rhsE += Dni * s_F[n, j, k, _E]
      r_rhsE += Dnj * s_G[i, n, k, _E]
      r_rhsE += Dnk * s_H[i, j, n, _E]
    end

    MJI = vgeo[i, j, k, _MJI, e]

    rhs[i, j, k, _U, e] += MJI * r_rhsU
    rhs[i, j, k, _V, e] += MJI * r_rhsV
    rhs[i, j, k, _W, e] += MJI * r_rhsW
    rhs[i, j, k, _ρ, e] += MJI * r_rhsρ
    rhs[i, j, k, _E, e] += MJI * r_rhsE
  end

  nothing
end

function volumerhs_trigpi!(::Val{N}, rhs, Q, vgeo, D, nelem, t) where {N}
  Nq = N + 1

  s_D = @cuStaticSharedMem eltype(D) (Nq, Nq)
  s_F = @cuStaticSharedMem eltype(Q) (Nq, Nq, Nq, _nstate)
  s_G = @cuStaticSharedMem eltype(Q) (Nq, Nq, Nq, _nstate)
  s_H = @cuStaticSharedMem eltype(Q) (Nq, Nq, Nq, _nstate)

  e = blockIdx().x
  k = threadIdx().z
  j = threadIdx().y
  i = threadIdx().x

  @inbounds begin
    # fetch D into shared
    k == 1 && (s_D[i, j] = D[i, j])

    # Load values will need into registers
    MJ = vgeo[i, j, k, _MJ, e]
    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
    x, y, z = vgeo[i,j,k,_x,e], vgeo[i,j,k,_y,e], vgeo[i,j,k,_z,e]

    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ))

    ρinv = 1 / ρ

    fluxρ_x = U
    fluxU_x = ρinv * U * U + P
    fluxV_x = ρinv * U * V
    fluxW_x = ρinv * U * W
    fluxE_x = ρinv * U * (E + P)

    fluxρ_y = V
    fluxU_y = ρinv * V * U
    fluxV_y = ρinv * V * V + P
    fluxW_y = ρinv * V * W
    fluxE_y = ρinv * V * (E + P)

    fluxρ_z = W
    fluxU_z = ρinv * W * U
    fluxV_z = ρinv * W * V
    fluxW_z = ρinv * W * W + P
    fluxE_z = ρinv * W * (E + P)

    s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
    s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
    s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
    s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
    s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

    s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
    s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
    s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
    s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
    s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

    s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
    s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
    s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
    s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
    s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

    r_rhsρ = pi*(-CUDAnative.sinpi(t)*CUDAnative.sinpi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 2*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y) +
                 CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 3*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(z) +
                 2*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 3*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 3*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z))
    r_rhsU = pi*(-10*CUDAnative.sinpi(t)*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 15*CUDAnative.sinpi(t)*CUDAnative.sinpi(x)*CUDAnative.cospi(z) -
                 15*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 10*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 5*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 30*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(z) -
                 3*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 15*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y) +
                 9*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 15*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 6*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y) +
                 18*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 2*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(z))*CUDAnative.cospi(y)/5
    r_rhsV = pi*(-10*CUDAnative.sinpi(t)*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 15*CUDAnative.sinpi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 3*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 9*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 10*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 5*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 6*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y) -
                 18*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 15*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y) +
                 15*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 15*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 2*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(z) +
                 30*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z))*CUDAnative.sinpi(x)/5
    r_rhsW = pi*(-10*CUDAnative.sinpi(t)*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 15*CUDAnative.sinpi(t) -
                 15*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.sinpi((x + y))*CUDAnative.cospi(t)*CUDAnative.cospi(z) +
                 15*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(z) +
                 36*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 18*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 4*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y) +
                 18*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 4*CUDAnative.cospi(t)*CUDAnative.cospi(y) +
                 30*CUDAnative.cospi(t)*CUDAnative.cospi(z)*CUDAnative.cospi((x + y)) -
                 2*CUDAnative.cospi(t))*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(y)/5
    r_rhsE = pi*(3*(1 - CUDAnative.cos(2*pi*x))*
                 (1 - CUDAnative.cos(4*pi*z))*(CUDAnative.cos(2*pi*t) + 1)*
                 (CUDAnative.cos(2*pi*y) + 1)/512 - 5*CUDAnative.sinpi(t)*
                 CUDAnative.sinpi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 4*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 8*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y) -
                 2*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 9*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 18*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 4*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 3*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 8*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 6*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 14*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 9*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 7*CUDAnative.sinpi(x)*CUDAnative.sinpi(z)*CUDAnative.cospi(t)*CUDAnative.cospi(y) -
                 18*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 7*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) -
                 700*CUDAnative.sinpi(x)*CUDAnative.sinpi(y)*CUDAnative.cospi(t)*CUDAnative.cospi(z) +
                 14*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 700*CUDAnative.sinpi(x)*CUDAnative.cospi(t)*CUDAnative.cospi(y)*CUDAnative.cospi(z) +
                 700*CUDAnative.cospi(t)*CUDAnative.cospi(x)*CUDAnative.cospi(y)*CUDAnative.cospi(z))/5

    sync_threads()

    # loop of ξ-grid lines
    @unroll for n = 1:Nq
      Dni = s_D[n, i]
      Dnj = s_D[n, j]
      Dnk = s_D[n, k]

      r_rhsρ += Dni * s_F[n, j, k, _ρ]
      r_rhsρ += Dnj * s_G[i, n, k, _ρ]
      r_rhsρ += Dnk * s_H[i, j, n, _ρ]

      r_rhsU += Dni * s_F[n, j, k, _U]
      r_rhsU += Dnj * s_G[i, n, k, _U]
      r_rhsU += Dnk * s_H[i, j, n, _U]

      r_rhsV += Dni * s_F[n, j, k, _V]
      r_rhsV += Dnj * s_G[i, n, k, _V]
      r_rhsV += Dnk * s_H[i, j, n, _V]

      r_rhsW += Dni * s_F[n, j, k, _W]
      r_rhsW += Dnj * s_G[i, n, k, _W]
      r_rhsW += Dnk * s_H[i, j, n, _W]

      r_rhsE += Dni * s_F[n, j, k, _E]
      r_rhsE += Dnj * s_G[i, n, k, _E]
      r_rhsE += Dnk * s_H[i, j, n, _E]
    end

    MJI = vgeo[i, j, k, _MJI, e]

    rhs[i, j, k, _U, e] += MJI * r_rhsU
    rhs[i, j, k, _V, e] += MJI * r_rhsV
    rhs[i, j, k, _W, e] += MJI * r_rhsW
    rhs[i, j, k, _ρ, e] += MJI * r_rhsρ
    rhs[i, j, k, _E, e] += MJI * r_rhsE
  end

  nothing
end

function main()
    N = 4
    nelem = 4000
    ntrials = 1
    DFloat = Float32

    rnd = MersenneTwister(0)

    Nq = N + 1
    Q = 1 .+ CuArray(rand(rnd, DFloat, Nq, Nq, Nq, _nstate, nelem))
    Q[:, :, :, _E, :] .+= 20
    vgeo = CuArray(rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem))

    # Make sure the entries of the mass matrix satisfy the inverse relation
    vgeo[:, :, :, _MJ, :] .+= 3
    vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

    D = CuArray(rand(rnd, DFloat, Nq, Nq))

    rhs = CuArray(zeros(DFloat, Nq, Nq, Nq, _nstate, nelem))

    t = DFloat(1)
    @cuda(threads=(N+1, N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs!(Val(N), rhs, Q, vgeo, D, nelem, t))

    CUDAdrv.@profile  begin
        @cuda(threads=(N+1, N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs!(Val(N), rhs, Q, vgeo, D, nelem, t))
    end

    @cuda(threads=(N+1, N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs_trigpi!(Val(N), rhs, Q, vgeo, D, nelem, t))

    CUDAdrv.@profile  begin
        @cuda(threads=(N+1, N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs_trigpi!(Val(N), rhs, Q, vgeo, D, nelem, t))
    end

    #=
    CUDAnative.@device_code dir="jlir" begin
        @cuda(threads=(N+1, N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs!(Val(N), rhs, Q, vgeo, D, nelem, t))
    end
    =#

    nothing
end

main()
