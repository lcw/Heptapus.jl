using LinearAlgebra, Logging, Random
using CUDAdrv
using CUDAnative
using CuArrays
using GPUifyLoops
using StaticArrays
using ArgParse

@inline function GPUifyLoops.Cassette.overdub(::GPUifyLoops.Ctx, ::typeof(*), a::Float32, b::Float32)
  Base.llvmcall("""
           %x = fmul contract float %0, %1
           ret float %x
           """, Float32, Tuple{Float32, Float32}, a, b)
end

@inline function GPUifyLoops.Cassette.overdub(::GPUifyLoops.Ctx, ::typeof(+), a::Float32, b::Float32)
  Base.llvmcall("""
           %x = fadd contract float %0, %1
           ret float %x
           """, Float32, Tuple{Float32, Float32}, a, b)
end

@inline function GPUifyLoops.Cassette.overdub(::GPUifyLoops.Ctx, ::typeof(-), a::Float32, b::Float32)
  Base.llvmcall("""
           %x = fsub contract float %0, %1
           ret float %x
           """, Float32, Tuple{Float32, Float32}, a, b)
end

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 5
const _ρ, _U, _V, _W, _E = 1:_nstate
const stateid = (ρ = _ρ, U = _U, V = _V, W = _W, E = _E)

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
_x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                x = _x,   y = _y,   z = _z)

Base.@irrational grav 9.81 BigFloat(9.81)
Base.@irrational R_d 287.0024093890231 BigFloat(287.0024093890231)
Base.@irrational cp_d 1004.5084328615809 BigFloat(1004.5084328615809)
Base.@irrational cv_d 717.5060234725578 BigFloat(717.5060234725578)
Base.@irrational gamma_d 1.4 BigFloat(1.4)
Base.@irrational gdm1 0.4 BigFloat(0.4)
# }}}

# {{{ volumerhs_v2!
function volumerhs_v2!(::Val{3}, ::Val{N}, ::Val{nmoist},
                       ::Val{ntrace}, rhs, Q, vgeo, gravity, D,
                       nelem) where {N, nmoist, ntrace}
    DFloat = eltype(Q)

    nvar = _nstate + nmoist + ntrace

    Nq = N + 1

    s_D = @shmem DFloat (Nq, Nq)
    s_F = @shmem DFloat (Nq, Nq, Nq, _nstate)
    s_G = @shmem DFloat (Nq, Nq, Nq, _nstate)
    s_H = @shmem DFloat (Nq, Nq, Nq, _nstate)
    l_ρinv = @shmem DFloat (Nq, Nq, Nq) 

    @inbounds @loop for e in (1:nelem; blockIdx().x)
        @loop for k in (1:Nq; threadIdx().z)
            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)

                    if k == 1
                        s_D[i, j] = D[i, j]
                    end

                    # Load values will need into registers
                    MJ = vgeo[i, j, k, _MJ, e]
                    ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                    ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                    ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
                    z = vgeo[i,j,k,_z,e]

                    U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
                    ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

                    P = gdm1*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)

                    #          l_ρinv[i, j, k] = ρinv = 1 / ρ
                    l_ρinv[i, j, k] = ρinv = 1 / ρ
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
                end
            end
        end

        @synchronize

        @loop for k in (1:Nq; threadIdx().z)
            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)
                    # TODO: Prefetch MJI and rhs

                    rhsU = rhsV = rhsW = rhsρ = rhsE = zero(DFloat)
                    MJI = vgeo[i, j, k, _MJI, e]

                    # buoyancy term
                    ρ = Q[i, j, k, _ρ, e]
                    rhsW -= ρ * gravity

                    # loop of ξ-grid lines
                    @unroll for n = 1:Nq
                        MJI_Dni = MJI * s_D[n, i]
                        MJI_Dnj = MJI * s_D[n, j]
                        MJI_Dnk = MJI * s_D[n, k]

                        rhsρ += MJI_Dni * s_F[n, j, k, _ρ]
                        rhsρ += MJI_Dnj * s_G[i, n, k, _ρ]
                        rhsρ += MJI_Dnk * s_H[i, j, n, _ρ]

                        rhsU += MJI_Dni * s_F[n, j, k, _U]
                        rhsU += MJI_Dnj * s_G[i, n, k, _U]
                        rhsU += MJI_Dnk * s_H[i, j, n, _U]

                        rhsV += MJI_Dni * s_F[n, j, k, _V]
                        rhsV += MJI_Dnj * s_G[i, n, k, _V]
                        rhsV += MJI_Dnk * s_H[i, j, n, _V]

                        rhsW += MJI_Dni * s_F[n, j, k, _W]
                        rhsW += MJI_Dnj * s_G[i, n, k, _W]
                        rhsW += MJI_Dnk * s_H[i, j, n, _W]

                        rhsE += MJI_Dni * s_F[n, j, k, _E]
                        rhsE += MJI_Dnj * s_G[i, n, k, _E]
                        rhsE += MJI_Dnk * s_H[i, j, n, _E]
                    end

                    rhs[i, j, k, _U, e] += rhsU
                    rhs[i, j, k, _V, e] += rhsV
                    rhs[i, j, k, _W, e] += rhsW
                    rhs[i, j, k, _ρ, e] += rhsρ
                    rhs[i, j, k, _E, e] += rhsE
                end
            end
        end

        # loop over moist variables
        # FIXME: Currently just passive advection
        # TODO: This should probably be unrolled by some factor
        rhsmoist = zero(eltype(rhs))
        @unroll for m = 1:nmoist
            s = _nstate + m

            @synchronize

            @loop for k in (1:Nq; threadIdx().z)
                @loop for j in (1:Nq; threadIdx().y)
                    @loop for i in (1:Nq; threadIdx().x)
                        MJ = vgeo[i, j, k, _MJ, e]
                        ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                        ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                        ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

                        Qmoist = Q[i, j, k, s, e]
                        U = Q[i, j, k, _U, e]
                        V = Q[i, j, k, _V, e]
                        W = Q[i, j, k, _W, e]

                        fx = U * l_ρinv[i, j, k] * Qmoist
                        fy = V * l_ρinv[i, j, k] * Qmoist
                        fz = W * l_ρinv[i, j, k] * Qmoist

                        s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
                        s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
                        s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
                    end
                end
            end

            @synchronize

            @loop for k in (1:Nq; threadIdx().z)
                @loop for j in (1:Nq; threadIdx().y)
                    @loop for i in (1:Nq; threadIdx().x)
                        # TODO: Prefetch MJI and rhs
                        MJI = vgeo[i, j, k, _MJI, e]

                        rhsmoist = zero(DFloat)
                        @unroll for n = 1:Nq
                            MJI_Dni = MJI * s_D[n, i]
                            MJI_Dnj = MJI * s_D[n, j]
                            MJI_Dnk = MJI * s_D[n, k]

                            rhsmoist += MJI_Dni * s_F[n, j, k, 1]
                            rhsmoist += MJI_Dnj * s_G[i, n, k, 1]
                            rhsmoist += MJI_Dnk * s_H[i, j, n, 1]
                        end
                        rhs[i, j, k, s, e] += rhsmoist
                    end
                end
            end
        end

        # Loop over trace variables
        # TODO: This should probably be unrolled by some factor
        rhstrace = zero(eltype(rhs))
        @unroll for m = 1:ntrace
            s = _nstate + nmoist + m

            @synchronize

            @loop for k in (1:Nq; threadIdx().z)
                @loop for j in (1:Nq; threadIdx().y)
                    @loop for i in (1:Nq; threadIdx().x)
                        MJ = vgeo[i, j, k, _MJ, e]
                        ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
                        ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
                        ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

                        Qtrace = Q[i, j, k, s, e]
                        U = Q[i, j, k, _U, e]
                        V = Q[i, j, k, _V, e]
                        W = Q[i, j, k, _W, e]

                        fx = U * l_ρinv[i, j, k] * Qtrace
                        fy = V * l_ρinv[i, j, k] * Qtrace
                        fz = W * l_ρinv[i, j, k] * Qtrace

                        s_F[i, j, k, 1] = MJ * (ξx * fx + ξy * fy + ξz * fz)
                        s_G[i, j, k, 1] = MJ * (ηx * fx + ηy * fy + ηz * fz)
                        s_H[i, j, k, 1] = MJ * (ζx * fx + ζy * fy + ζz * fz)
                    end
                end
            end

            @synchronize

            @loop for k in (1:Nq; threadIdx().z)
                @loop for j in (1:Nq; threadIdx().y)
                    @loop for i in (1:Nq; threadIdx().x)
                        # TODO: Prefetch MJI and rhs
                        MJI = vgeo[i, j, k, _MJI, e]

                        rhstrace = zero(DFloat)
                        @unroll for n = 1:Nq
                            MJI_Dni = MJI * s_D[n, i]
                            MJI_Dnj = MJI * s_D[n, j]
                            MJI_Dnk = MJI * s_D[n, k]

                            rhstrace += MJI_Dni * s_F[n, j, k, 1]
                            rhstrace += MJI_Dnj * s_G[i, n, k, 1]
                            rhstrace += MJI_Dnk * s_H[i, j, n, 1]
                        end
                        rhs[i, j, k, s, e] += rhstrace
                    end
                end
            end
        end
    end
    nothing
end
# }}}

# {{{ volumerhs_v3!
function volumerhs_v3!(::Val{3},
                       ::Val{N},
                       ::Val{nmoist},
                       ::Val{ntrace},
                       rhs,
                       Q,
                       vgeo,
                       gravity,
                       D,
                       nelem) where {N, nmoist, ntrace}
    nvar = _nstate + nmoist + ntrace

    Nq = N + 1

    s_D = @shmem eltype(D) (Nq, Nq)
    s_F = @shmem eltype(Q) (Nq, Nq, _nstate)
    s_G = @shmem eltype(Q) (Nq, Nq, _nstate)
    s_H = @shmem eltype(Q) (Nq, Nq, _nstate)

    r_rhsρ = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsU = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsV = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsW = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsE = @scratch eltype(rhs) (Nq, Nq, Nq) 2

    @inbounds @loop for e in (1:nelem; blockIdx().x)
        @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
                 for k in 1:Nq
                     r_rhsρ[k, i, j] = zero(eltype(rhs))
                     r_rhsU[k, i, j] = zero(eltype(rhs))
                     r_rhsV[k, i, j] = zero(eltype(rhs))
                     r_rhsW[k, i, j] = zero(eltype(rhs))
                     r_rhsE[k, i, j] = zero(eltype(rhs))
                end

                # fetch D into shared
                s_D[i, j] = D[i, j]
            end
        end

        @unroll for k in 1:Nq
            @synchronize
            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)

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
                    for n = 1:Nq
                        Dkn = s_D[k, n]

                        r_rhsρ[n,i,j] += Dkn * r_Hρ
                        r_rhsU[n,i,j] += Dkn * r_HU
                        r_rhsV[n,i,j] += Dkn * r_HV
                        r_rhsW[n,i,j] += Dkn * r_HW
                        r_rhsE[n,i,j] += Dkn * r_HE
                    end

                    r_rhsW[k,i,j] -= MJ * ρ * gravity
                end
            end

            @synchronize

            @loop for j in (1:Nq; threadIdx().y)
                @loop for i in (1:Nq; threadIdx().x)

                    # loop of ξ-grid lines
                    for n = 1:Nq
                        Dni = s_D[n, i]
                        Dnj = s_D[n, j]

                        r_rhsρ[k,i,j] += Dni * s_F[n, j, _ρ]
                        r_rhsρ[k,i,j] += Dnj * s_G[i, n, _ρ]

                        r_rhsU[k,i,j] += Dni * s_F[n, j, _U]
                        r_rhsU[k,i,j] += Dnj * s_G[i, n, _U]

                        r_rhsV[k,i,j] += Dni * s_F[n, j, _V]
                        r_rhsV[k,i,j] += Dnj * s_G[i, n, _V]

                        r_rhsW[k,i,j] += Dni * s_F[n, j, _W]
                        r_rhsW[k,i,j] += Dnj * s_G[i, n, _W]

                        r_rhsE[k,i,j] += Dni * s_F[n, j, _E]
                        r_rhsE[k,i,j] += Dnj * s_G[i, n, _E]
                    end 
                end 
            end
        end # k

        @loop for j in (1:Nq; threadIdx().y)
            @loop for i in (1:Nq; threadIdx().x)
                for k in 1:Nq
                    MJI = vgeo[i, j, k, _MJI, e]

                    rhs[i, j, k, _U, e] += MJI*r_rhsU[k, i, j]
                    rhs[i, j, k, _V, e] += MJI*r_rhsV[k, i, j]
                    rhs[i, j, k, _W, e] += MJI*r_rhsW[k, i, j]
                    rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k, i, j]
                    rhs[i, j, k, _E, e] += MJI*r_rhsE[k, i, j]
                end
            end
        end
    end
    nothing
end
# }}}

# {{{ volumerhs_v3_1
function volumerhs_v3_1!(::Val{3},
                         ::Val{N},
                         ::Val{nmoist},
                         ::Val{ntrace},
                         rhs,
                         Q,
                         vgeo,
                         gravity,
                         D,
                         nelem) where {N, nmoist, ntrace}

    nvar = _nstate + nmoist + ntrace

    Nq = N + 1

    s_D = @shmem eltype(D) (Nq, Nq)
    s_F = @shmem eltype(Q) (Nq, Nq, _nstate)
    s_G = @shmem eltype(Q) (Nq, Nq, _nstate)
    s_H = @shmem eltype(Q) (Nq, Nq, _nstate)

    r_rhsρ = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsU = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsV = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsW = @scratch eltype(rhs) (Nq, Nq, Nq) 2
    r_rhsE = @scratch eltype(rhs) (Nq, Nq, Nq) 2

    @inbounds for e = blockIdx().x
        for j = threadIdx().y
            for i = threadIdx().x
                @unroll for k in 1:Nq
                    r_rhsρ[k, i, j] = zero(eltype(rhs))
                    r_rhsU[k, i, j] = zero(eltype(rhs))
                    r_rhsV[k, i, j] = zero(eltype(rhs))
                    r_rhsW[k, i, j] = zero(eltype(rhs))
                    r_rhsE[k, i, j] = zero(eltype(rhs))
                end

                # fetch D into shared
                s_D[i, j] = D[i, j]
            end
        end

        @unroll for k in 1:Nq
            @synchronize
            for j = threadIdx().y
                for i = threadIdx().x

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
                        Dkn = s_D[k, n]

                        r_rhsρ[n,i,j] += Dkn * r_Hρ
                        r_rhsU[n,i,j] += Dkn * r_HU
                        r_rhsV[n,i,j] += Dkn * r_HV
                        r_rhsW[n,i,j] += Dkn * r_HW
                        r_rhsE[n,i,j] += Dkn * r_HE
                    end

                    r_rhsW[k,i,j] -= MJ * ρ * gravity
                end
            end

            @synchronize

            for j = threadIdx().y
                for i = threadIdx().x

                    # loop of ξ-grid lines
                    @unroll for n = 1:Nq
                        Dni = s_D[n, i]
                        Dnj = s_D[n, j]

                        r_rhsρ[k,i,j] += Dni * s_F[n, j, _ρ]
                        r_rhsρ[k,i,j] += Dnj * s_G[i, n, _ρ]

                        r_rhsU[k,i,j] += Dni * s_F[n, j, _U]
                        r_rhsU[k,i,j] += Dnj * s_G[i, n, _U]

                        r_rhsV[k,i,j] += Dni * s_F[n, j, _V]
                        r_rhsV[k,i,j] += Dnj * s_G[i, n, _V]

                        r_rhsW[k,i,j] += Dni * s_F[n, j, _W]
                        r_rhsW[k,i,j] += Dnj * s_G[i, n, _W]

                        r_rhsE[k,i,j] += Dni * s_F[n, j, _E]
                        r_rhsE[k,i,j] += Dnj * s_G[i, n, _E]
                    end
                end
            end
        end # k

        for j = threadIdx().y
            for i = threadIdx().x
                @unroll for k in 1:Nq
                    MJI = vgeo[i, j, k, _MJI, e]

                    rhs[i, j, k, _U, e] += MJI*r_rhsU[k, i, j]
                    rhs[i, j, k, _V, e] += MJI*r_rhsV[k, i, j]
                    rhs[i, j, k, _W, e] += MJI*r_rhsW[k, i, j]
                    rhs[i, j, k, _ρ, e] += MJI*r_rhsρ[k, i, j]
                    rhs[i, j, k, _E, e] += MJI*r_rhsE[k, i, j]
                end
            end
        end
    end
    nothing
end
# }}}

# {{{ main
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--Float64"
            help = "Run with Float64 data (otherwise run with Float32)"
            action = :store_true
        "--ntrials"
            help = "number trials for each kernel"
            arg_type = Int
            default = 100
        "N"
            help = "polynomial order"
            required = true
            arg_type = Int
        "nelem"
            help = "number of elements"
            required = true
            arg_type = Int
    end

    return parse_args(s)
end



function main()
    CUDAnative.timings!()

    parsed_args = parse_commandline()

    N = parsed_args["N"]
    nelem = parsed_args["nelem"]
    ntrials = parsed_args["ntrials"]
    DFloat = parsed_args["Float64"] ? Float64 : Float32

    rnd = MersenneTwister(0)
    nmoist = 0
    ntrace = 0
    nvar = _nstate + nmoist + ntrace

    Nq = N + 1
    Q = 1 .+ CuArray(rand(rnd, DFloat, Nq, Nq, Nq, nvar, nelem))
    Q[:, :, :, _E, :] .+= 20
    vgeo = CuArray(rand(rnd, DFloat, Nq, Nq, Nq, _nvgeo, nelem))

    # Make sure the entries of the mass matrix satisfy the inverse relation
    vgeo[:, :, :, _MJ, :] .+= 3
    vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

    D = CuArray(rand(rnd, DFloat, Nq, Nq))

    rhs = CuArray(zeros(DFloat, Nq, Nq, Nq, nvar, nelem))

    #
    # volumerhs_v2!
    #
    fill!(rhs, 0)
    @launch(CUDA(), threads=(N+1, N+1, N+1), blocks=nelem,
          volumerhs_v2!(Val(3), Val(N), Val(nmoist), Val(ntrace),
                        rhs, Q, vgeo, DFloat(grav), D, nelem))
    norm_v2 = norm(rhs)
    CUDAdrv.@profile for _ = 1:ntrials
         @launch(CUDA(), threads=(N+1, N+1, N+1), blocks=nelem,
               volumerhs_v2!(Val(3), Val(N), Val(nmoist), Val(ntrace), rhs,
                             Q, vgeo, DFloat(grav), D, nelem))
    end

    #
    # volumerhs_v3!
    #
    fill!(rhs, 0)
    @launch(CUDA(), threads=(N+1, N+1), blocks=nelem,
            volumerhs_v3!(Val(3),
                       Val(N),
                       Val(nmoist),
                       Val(ntrace),
                       rhs,
                       Q,
                       vgeo,
                       DFloat(grav),
                       D,
                       nelem))
    norm_v3 = norm(rhs)
    CUDAdrv.@profile for _ = 1:ntrials
        @launch(CUDA(), threads=(N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs_v3!(Val(3), Val(N), Val(nmoist), Val(ntrace), rhs,
                            Q, vgeo, DFloat(grav), D, nelem))
    end

    #
    # volumerhs_v3_1!
    #
    fill!(rhs, 0)
    @launch(CUDA(), threads=(N+1, N+1), blocks=nelem, maxregs=255,
          volumerhs_v3_1!(Val(3), Val(N), Val(nmoist), Val(ntrace),
                          rhs, Q, vgeo, DFloat(grav), D, nelem))
    norm_v3_1 = norm(rhs)
    CUDAdrv.@profile for _ = 1:ntrials
        @launch(CUDA(), threads=(N+1, N+1), blocks=nelem, maxregs=255,
              volumerhs_v3_1!(Val(3), Val(N), Val(nmoist), Val(ntrace),
                              rhs, Q, vgeo, DFloat(grav), D, nelem))
    end

    @assert norm_v2 ≈ norm_v3
    @assert norm_v2 ≈ norm_v3_1

    CUDAnative.timings()

    nothing
end
# }}}

main()
