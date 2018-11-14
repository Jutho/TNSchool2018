using LinearAlgebra, TensorOperations, KrylovKit

"""
    randisometry([T=Float64], dims...)

Construct a random isometry
"""
randisometry(T, d1, d2) = d1 >= d2 ? Matrix(qr!(randn(T, d1, d2)).Q) : Matrix(lq!(randn(T, d1, d2)).Q)
randisometry(d1, d2) = randisometry(Float64, d1, d2)
randisometry(dims::Dims{2}) = randisometry(dims[1], dims[2])
randisometry(T, dims::Dims{2}) = randisometry(T, dims[1], dims[2])

"""
    randmps(physdims::NTuple{N,Int}, Dmax::Int, [T::Type{<:Number} = Float64])
    randmps(N::Int, d::Int, Dmax::Int, [T::Type{<:Number} = Float64])

Construct a random right canonical MPS for a system with `N`, where site `n` has local Hilbert
space dimension `physdims[n]` (first method) or `d` (second method), and the maximal bond
dimension is `Dmax`. Entries of the MPS tensors will be of type `T`, defaulting to `Float64`.
"""
function randmps(physdims::Dims{N}, Dmax::Int, T::Type{<:Number} = Float64) where {N}
    bonddims = Vector{Int}(undef, N+1)
    bonddims[1] = 1
    bonddims[N+1] = 1
    Nhalf = div(N,2)
    for n = 2:N
        bonddims[n] = min(Dmax, bonddims[n-1]*physdims[n-1])
    end
    for n = N:-1:1
        bonddims[n] = min(bonddims[n], bonddims[n+1]*physdims[n])
    end

    As = Vector{Any}(undef, N)
    for n = 1:N
        d = physdims[n]
        Dl = bonddims[n]
        Dr = bonddims[n+1]
        As[n] = reshape(randisometry(T, Dl, d*Dr), (Dl, d, Dr))
    end
    return As
end
randmps(N::Int, d::Int, Dmax::Int, T = Float64) = randmps(ntuple(n->d, N), Dmax, T)

"""
    entanglemententropy(A)

For a list of tensors `A` representing a right orthonormalized MPS, compute the entanglement
entropy for a bipartite cut for every bond
"""
function entanglemententropy(A)
    N = length(A)
    entropy = Vector{Float64}(undef, N-1)

    A1 = A[1]
    U, S, V = svdtrunc(reshape(A1, size(A1, 1)*size(A1,2), :))
    schmidtcoeffs = diag(S)
    entropy[1] = sum(-schmidtcoeffs.^2 .* log.(schmidtcoeffs.^2))
    for k = 2:N-1
        Ak = S*V*reshape(A[k], size(A[k], 1), :)
        U, S, V = svdtrunc(reshape(Ak, size(Ak, 1)*size(A[k],2), :))
        schmidtcoeffs = diag(S)
        entropy[k] = sum(-schmidtcoeffs.^2 .* log.(schmidtcoeffs.^2))
    end
    return entropy
end

"""
    measure1siteoperator(A, O)

For a list of tensors `A` representing a right orthonormalized MPS, compute the local expectation
value of a one-site operator O for every site.
"""
function measure1siteoperator(A, O)
    N = length(A)
    ρ = ones(eltype(A[1]), 1, 1)
    expval = Vector{ComplexF64}(undef, N)
    for k = 1:N
        @tensor v = scalar(ρ[a,b]*A[k][b,s,c]*O[s',s]*conj(A[k][a,s',c]))
        expval[k] = v
        @tensor ρ[a,b] := ρ[a',b']*A[k][b',s,b]*conj(A[k][a',s,a])
    end
    return expval
end


"""
    applyH2(AAC, FL, FR, M1, M2)

Apply the effective Hamiltonian on the two-site center tensor `AAC`, by contracting with the
left and right environment `FL` and `FR` and two MPO tensors `M1` and `M2`
"""
function applyH2(AAC, FL, FR, M1, M2)
    # TODO
end

"""
    applyH1(AC, FL, FR, M)

Apply the effective Hamiltonian on the center tensor `AC`, by contracting with the left and right
environment `FL` and `FR` and the MPO tensor `M`
"""
function applyH1(AC, FL, FR, M)
    # TODO
end
"""
    applyH0(C, FL, FR)

Apply the effective Hamiltonian on the bond matrix C, by contracting with the left and right
environment `FL` and `FR`
"""
function applyH0(C, FL, FR)
    # TODO
end
"""
    updateleftenv(A, M, FR)

Compute the left environment tensor for the next site, by contracting to the left environment
`FL` of the current site an extra MPO tensor `M` and two MPS tensors `A` and `conj(A)``.
"""
function updateleftenv(A, M, FL)
    # TODO
end
"""
    updaterightenv(A, M, FR)

Compute the right environment tensor for the previous site, by contracting to the right environment
`FR` of the current site, an extra MPO tensor `M` and two MPS tensors `A` and `conj(A)``.
"""
function updaterightenv(A, M, FR)
    # TODO
end

"""
    dmrg1sweep!(A, H [, F; kwargs...])

Run one sweep (left to right and back) of the one-site DMRG algorithm, updates `A` and `F` in
place, returns energy `E`, MPS tensors `A` and environments `F`. Assumes A starts in right
canonical form.
"""
function dmrg1sweep!(A, H, F = nothing; verbose = true, kwargs...)
    N = length(A)

    if F == nothing
        F = Vector{Any}(undef, N+2)
        F[1] = fill!(similar(H[1], (1,1,1)), 1)
        F[N+2] = fill!(similar(H[1], (1,1,1)), 1)
        for k = N:-1:1
            F[k+1] = updaterightenv(A[k], M[k], F[k+2])
        end
    end

    AC = A[1]
    for k = 1:N-1
        Es, ACs, info = eigsolve(x->applyH1(x, F[k], F[k+2], M[k]), AC, 1, :SR; ishermitian = true, kwargs...)
        AC = ACs[1]
        E = Es[1]

        verbose && println("Sweep L2R: site $k -> energy $E")

        AL, C = qr(reshape(AC, size(AC,1)*size(AC,2), :))
        A[k] = reshape(Matrix(AL), size(AC))
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        @tensor AC[-1,-2,-3] := C[-1,1] * A[k+1][1,-2,-3]
    end
    k = N
    Es, ACs, info = eigsolve(x->applyH1(x, F[k], F[k+2], M[k]), AC, 1, :SR; ishermitian = true, kwargs...)
    AC = ACs[1]
    E = Es[1]
    verbose && println("Sweep L2R: site $k -> energy $E")
    for k = N-1:-1:1
        C, AR = lq(reshape(AC, size(AC,1), :))
        # it's actually better to do qr of transpose and transpose back

        A[k+1] = reshape(Matrix(AR), size(AC))
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        @tensor AC[:] := A[k][-1,-2,1] * C[1,-3]
        Es, ACs, info = eigsolve(x->applyH1(x, F[k], F[k+2], M[k]), AC, 1, :SR; ishermitian = true, kwargs...)
        AC = ACs[1]
        E = Es[1]
        verbose && println("Sweep R2L: site $k -> energy $E")
    end
    A[1] = AC
    return E, A, F
end

"""
    tdvp1sweep(dt, A, H [, F; kwargs...])

Run one sweep (left to right and back) of the one-site TDVP algorithm, updates `A` and `F` in
place, returns MPS tensors `A` and environments `F`. Assumes A starts in right
canonical form.
"""
function tdvp1sweep!(dt, A, H, F = nothing; verbose = true, kwargs...)
    N = length(A)

    if F == nothing
        F = Vector{Any}(undef, N+2)
        F[1] = fill!(similar(H[1], (1,1,1)), 1)
        F[N+2] = fill!(similar(H[1], (1,1,1)), 1)
        for k = N:-1:1
            F[k+1] = updaterightenv(A[k], M[k], F[k+2])
        end
    end

    AC = A[1]
    for k = 1:N-1
        AC, info = exponentiate(x->applyH1(x, F[k], F[k+2], M[k]), -im*dt, AC; ishermitian = true, kwargs...)

        if verbose
            E = dot(AC, applyH1(AC, F[k], F[k+2], M[k]))
            println("Sweep L2R: AC site $k -> energy $E")
        end

        AL, C = qr(reshape(AC, size(AC,1)*size(AC,2), :))
        A[k] = reshape(Matrix(AL), size(AC))
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        # TODO: backward evolution of C

        if verbose
            E = dot(C, applyH0(C, F[k+1], F[k+2]))
            println("Sweep L2R: C between site $k and $(k+1) -> energy $E")
        end

        @tensor AC[-1,-2,-3] := C[-1,1] * A[k+1][1,-2,-3]
    end
    k = N
    AC, info = exponentiate(x->applyH1(x, F[k], F[k+2], M[k]), -im*dt, AC; ishermitian = true, kwargs...)

    if verbose
        E = dot(AC, applyH1(AC, F[k], F[k+2], M[k]))
        println("Sweep L2R: AC site $k -> energy $E")
    end

    for k = N-1:-1:1
        C, AR = lq(reshape(AC, size(AC,1), :))
        # it's actually better to do qr of transpose and transpose back

        A[k+1] = reshape(Matrix(AR), size(AC))
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        # TODO: backward evolution of C

        if verbose
            E = dot(C, applyH0(C, F[k+1], F[k+2]))
            println("Sweep R2L: C between site $k and $(k+1) -> energy $E")
        end

        @tensor AC[:] := A[k][-1,-2,1] * C[1,-3]
        AC, info = exponentiate(x->applyH1(x, F[k], F[k+2], M[k]), -im*dt, AC; ishermitian = true, kwargs...)

        if verbose
            E = dot(AC, applyH1(AC, F[k], F[k+2], M[k]))
            println("Sweep R2L: AC site $k -> energy $E")
        end
    end
    A[1] = AC
    return A, F
end

"""
    U, S, Vd = svdtrunc(A; truncdim = max(size(A)...), truncerr = 0.)

Perform a truncated SVD, with maximum number of singular values to keep equal to `truncdim`
or truncating any singular values smaller than `truncerr`. If both options are provided, the
smallest number of singular values will be kept.

Unlike the SVD in Julia, this returns matrix U, a diagonal matrix (not a vector) S, and
Vt such that A ≈ U * S * Vt
"""
function svdtrunc(A; truncdim = max(size(A)...), truncerr = 0.)
    F = svd(A)
    d = min(truncdim, count(F.S .>= truncerr))
    return F.U[:,1:d], diagm(0=>F.S[1:d]), F.Vt[1:d, :]
end

"""
    dmrg2sweep(A, H, [F]; verbose = true, truncdim = 200, truncerr = 1e-6)

Run one sweep (left to right and back) of the two-site DMRG algorithm, thereby truncating
the bond dimension to obtain truncation error `truncerr` or to maximum size `truncdim`. Updates
`A` and `F` in place, returns energy `E`, MPS tensors `A` and environments `F`. Assumes A starts
in right canonical form.
"""
function dmrg2sweep!(A, H, F = nothing; verbose = true, truncdim = 200, truncerr = 1e-6, kwargs...)
    N = length(A)

    if F == nothing
        F = Vector{Any}(undef, N+2)
        F[1] = fill!(similar(H[1], (1,1,1)), 1)
        F[N+2] = fill!(similar(H[1], (1,1,1)), 1)
        for k = N:-1:1
            F[k+1] = updaterightenv(A[k], M[k], F[k+2])
        end
    end

    AC = A[1]
    for k = 1:N-2
        @tensor AAC[-1,-2,-3,-4] := AC[-1,-2,1]*A[k+1][1,-3,-4]
        Es, AACs, info = eigsolve(x->applyH2(x, F[k], F[k+3], M[k], M[k+1]), AAC, 1, :SR; ishermitian = true, kwargs...)
        AAC = AACs[1]
        E = Es[1]

        verbose && println("Sweep L2R: site $(k:k+1) -> energy $E")

        AL, S, V = svdtrunc(reshape(AAC, size(AAC,1)*size(AAC,2), :); truncdim = truncdim, truncerr = truncerr)
        A[k] = reshape(AL, size(AC, 1), size(AC, 2), :)
        F[k+1] = updateleftenv(A[k], M[k], F[k])

        AC = reshape(S*V, size(S,1), size(A[k+1], 2), size(A[k+1], 3))
    end

    k = N-1
    @tensor AAC[-1,-2,-3,-4] := AC[-1,-2,1]*A[k+1][1,-3,-4]
    Es, AACs, info = eigsolve(x->applyH2(x, F[k], F[k+3], M[k], M[k+1]), AAC, 1, :SR; ishermitian = true, kwargs...)
    AAC = AACs[1]
    E = Es[1]
    verbose && println("Sweep L2R: site $(k:k+1) -> energy $E")

    for k = N-1:-1:2
        U, S, AR = svdtrunc(reshape(AAC, size(AAC,1)*size(AAC,2), :); truncdim = truncdim, truncerr = truncerr)

        A[k+1] = reshape(AR, size(AR, 1), size(AAC, 3), size(AAC, 4))
        F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

        AC = reshape(U*S, size(AAC,1), size(AAC,2), size(S,2))
        @tensor AAC[:] := A[k-1][-1,-2,1] * AC[1,-3,-4]

        Es, AACs, info = eigsolve(x->applyH2(x, F[k-1], F[k+2], M[k-1], M[k]), AAC, 1, :SR; ishermitian = true, kwargs...)
        AAC = AACs[1]
        E = Es[1]
        verbose && println("Sweep R2L: site $(k-1:k) -> energy $E")
    end
    k = 1
    U, S, AR = svdtrunc(reshape(AAC, size(AAC,1)*size(AAC,2), :); truncdim = truncdim, truncerr = truncerr)

    A[k+1] = reshape(AR, size(AR, 1), size(AAC, 3), size(AAC, 4))
    F[k+2] = updaterightenv(A[k+1], M[k+1], F[k+3])

    AC = reshape(U*S, size(AAC,1), size(AAC,2), size(S,2))
    A[1] = AC
    return E, A, F
end
