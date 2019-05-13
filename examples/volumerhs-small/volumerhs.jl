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

import Base: +, *, -

+(a::Float64, b::Float64) = add_float_contract(a, b)
+(a::Float32, b::Float32) = add_float_contract(a, b)
*(a::Float64, b::Float64) = mul_float_contract(a, b)
*(a::Float32, b::Float32) = mul_float_contract(a, b)
-(a::Float64, b::Float64) = sub_float_contract(a, b)
-(a::Float32, b::Float32) = sub_float_contract(a, b)

# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
       _x, _y, _z = 1:_nvgeo

Base.@irrational grav 9.81 BigFloat(9.81)
Base.@irrational gdm1 0.4 BigFloat(0.4)

function volumerhs!(::Val{N}, rhs, Q, vgeo, gravity, D, nelem) where {N}
    Nq = N + 1

    s_D = @cuStaticSharedMem eltype(D) (Nq, Nq)
    s_F = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)
    s_G = @cuStaticSharedMem eltype(Q) (Nq, Nq, _nstate)

    r_rhsρ = MArray{Tuple{Nq},eltype(rhs)}(undef)
    r_rhsU = MArray{Tuple{Nq},eltype(rhs)}(undef)
    r_rhsV = MArray{Tuple{Nq},eltype(rhs)}(undef)
    r_rhsW = MArray{Tuple{Nq},eltype(rhs)}(undef)
    r_rhsE = MArray{Tuple{Nq},eltype(rhs)}(undef)

    e = blockIdx().x
    j = threadIdx().y
    i = threadIdx().x

    # fetch D into shared
    @inbounds s_D[i, j] = D[i, j]

    @inbounds @unroll for k in 1:Nq
        r_rhsρ[k] = zero(eltype(rhs))
        r_rhsU[k] = zero(eltype(rhs))
        r_rhsV[k] = zero(eltype(rhs))
        r_rhsW[k] = zero(eltype(rhs))
        r_rhsE[k] = zero(eltype(rhs))
    end

    @inbounds @unroll for k in 1:Nq
        # Load values will need into registers
        MJ = vgeo[i, j, k, _MJ, e]
        ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
        ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
        ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
        z = vgeo[i,j,k,_z,e]

        U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
        ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

        P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

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

        sync_threads()

        s_F[i, j,  _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
        s_F[i, j,  _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
        s_F[i, j,  _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
        s_F[i, j,  _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
        s_F[i, j,  _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

        s_G[i, j,  _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
        s_G[i, j,  _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
        s_G[i, j,  _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
        s_G[i, j,  _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
        s_G[i, j,  _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

        r_Hρ = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
        r_HU = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
        r_HV = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
        r_HW = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
        r_HE = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

        # one shared access per 10 flops
        @unroll for n = 1:Nq
            Dnk = s_D[k, n]

            r_rhsρ[n] += Dnk * r_Hρ
            r_rhsU[n] += Dnk * r_HU
            r_rhsV[n] += Dnk * r_HV
            r_rhsW[n] += Dnk * r_HW
            r_rhsE[n] += Dnk * r_HE
        end

        r_rhsW[k] -= MJ * ρ * gravity

        sync_threads()

        # loop of ξ-grid lines
        @unroll for n = 1:Nq
            Dni = s_D[n, i]
            Dnj = s_D[n, j]

            r_rhsρ[k] += Dni * s_F[n, j, _ρ]
            r_rhsρ[k] += Dnj * s_G[i, n, _ρ]

            r_rhsU[k] += Dni * s_F[n, j, _U]
            r_rhsU[k] += Dnj * s_G[i, n, _U]

            r_rhsV[k] += Dni * s_F[n, j, _V]
            r_rhsV[k] += Dnj * s_G[i, n, _V]

            r_rhsW[k] += Dni * s_F[n, j, _W]
            r_rhsW[k] += Dnj * s_G[i, n, _W]

            r_rhsE[k] += Dni * s_F[n, j, _E]
            r_rhsE[k] += Dnj * s_G[i, n, _E]
        end
    end

    @inbounds @unroll for k in 1:Nq
        MJI = vgeo[i, j, k, _MJI, e]

        rhs[i, j, k, _U, e] += MJI*r_rhsU[k]
        rhs[i, j, k, _V, e] += MJI*r_rhsV[k]
        rhs[i, j, k, _W, e] += MJI*r_rhsW[k]
        rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k]
        rhs[i, j, k, _E, e] += MJI*r_rhsE[k]
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

    @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs!(Val(N), rhs, Q, vgeo, DFloat(grav), D, nelem))

    CUDAdrv.@profile  begin
        @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs!(Val(N), rhs, Q, vgeo, DFloat(grav), D, nelem))
    end

    CUDAnative.@device_code dir="jlir" begin
        @cuda(threads=(N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs!(Val(N), rhs, Q, vgeo, DFloat(grav), D, nelem))
    end

    nothing
end

main()
