using LinearAlgebra, TensorOperations, KrylovKit

safesign(x::Number) = iszero(x) ? one(x) : sign(x)
"""
    qrpos(A)

Returns a QR decomposition, i.e. an isometric `Q` and upper triangular `R` matrix, where `R`
is guaranteed to have positive diagonal elements.
"""
qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    return Q, R
end

"""
    lqpos(A)

Returns a LQ decomposition, i.e. a lower triangular `L` and isometric `Q` matrix, where `L`
is guaranteed to have positive diagonal elements.
"""
lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    return L, Q
end

"""
    leftorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a left-canonical MPS tensor `AL`, a gauge transform `C` and
a scalar factor `λ` such that ``λ AL^s C = C A^s``, where an initial guess for `C` can be
provided.
"""
function leftorth(A, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, kwargs...)
    λ2s, ρs, info = schursolve(C'*C, 1, :LM, Arnoldi(tol = tol, kwargs...)) do ρ
        @tensor ρE[a,b] := ρ[a',b']*A[b',s,b]*conj(A[a',s,a])
        return ρE
    end
    ρ = ρs[1] + ρs[1]'
    ρ ./= tr(ρ)
    C = cholesky!(ρ).U

    D, d, = size(A)
    Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    rmul!(R, 1/λ)
    while norm(C-R) > tol
        C = R
        Q, R = qrpos!(reshape(C*reshape(A, D, d*D), D*d, D))
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        rmul!(R, 1/λ)
    end
    C = R
    return AL, C, λ
end

"""
    rightorth(A, [C]; kwargs...)

Given an MPS tensor `A`, return a gauge transform C, a right-canonical MPS tensor `AR`, and
a scalar factor `λ` such that ``λ C AR^s = A^s C``, where an initial guess for `C` can be
provided.
"""
function rightorth(A, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, kwargs...)
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
    leftenv(A, M, FL)

Compute the left environment tensor for MPS A and MPO M, by finding the left fixed point
of A - M - conj(A) contracted along the physical dimension.
"""
function leftenv(A, M, FL = randn(eltype(A), size(A,1), size(M,1), size(A,1)); kwargs...)
    λs, FLs, info = schursolve(FL, 1, :LM, Arnoldi(;kwargs...)) do FL
        # TODO
    end
    return FLs[1], λs[1]
end
"""
    leftenv(A, M, FR)

Compute the left environment tensor for MPS A and MPO M, by finding the right fixed point
of A - M - conj(A) contracted along the physical dimension.
"""
function rightenv(A, M, FR = randn(eltype(A), size(A,1), size(M,1), size(A,1)); kwargs...)
    λs, FRs, info = schursolve(FR, 1, :LM, Arnoldi(;kwargs...)) do FR
        # TODO
    end
    return FRs[1], λs[1]
end
function vumps(A, M; verbose = true, tol = 1e-6, kwargs...)
    AL, = leftorth(A)
    C, AR = rightorth(A)

    FL, λL = leftenv(AL, M; kwargs...)
    FR, λR = rightenv(AR, M; kwargs...)

    verbose && println("Starting point has λ ≈ $λL ≈ $λR")

    AL, C, AR, λ, errL, errR = vumpsstep(AL, C, AR, FL, FR; tol = tol/10, kwargs...)
    # AL, C, = leftorth(AR, C; tol = tol/10, kwargs...)
    FL, λL = leftenv(AL, M, FL; tol = tol/10, kwargs...)
    FR, λR = rightenv(AR, M, FR; tol = tol/10, kwargs...)
    i = 1
    verbose && println("Step $i: λ ≈ $λ ≈ $λL ≈ $λR, err ≈ $errL ≈ $errR")
    while (errL+errR)/2 > tol
        AL, C, AR, λ, errL, errR = vumpsstep(AL, C, AR, FL, FR; tol = tol/10, kwargs...)
        # AL, C, = leftorth(AR, C; tol = tol/10, kwargs...)
        FL, λL = leftenv(AL, M, FL; tol = tol/10, kwargs...)
        FR, λR = rightenv(AR, M, FR; tol = tol/10, kwargs...)
        i += 1
        verbose && println("Step $i: λ ≈ $λ ≈ $λL ≈ $λR, err ≈ $errL ≈ $errR")
    end
    return λ, AL, AR, C, FL, FR
end

"""
    function vumpsstep(AL, C, AR, FL, FR; kwargs...)

Perform one step of the VUMPS algorithm
"""
function vumpsstep(AL, C, AR, FL, FR; kwargs...)
    # TODO: get an updated AC, C and λ
    QAC, RAC = qrpos(reshape(AC,(D*d, D)))
    QC, RC = qrpos(C)
    AL = reshape(QAC*QC', (D, d, D))
    errL = norm(RAC-RC)

    LAC, QAC = lqpos(reshape(AC,(D, d*D)))
    LC, QC = lqpos(C)

    AR = reshape(QC'*QAC, (D, d, D))
    errR = norm(LAC-LC)

    return AL, C, AR, λ, errL, errR
end
