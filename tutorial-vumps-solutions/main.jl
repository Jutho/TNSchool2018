using Revise
Revise.includet("mps.jl")

function statmechmpo(β, h, D)
    M = zeros(D,D,D,D)
    for i = 1:D
        M[i,i,i,i] = 1
    end
    X = zeros(D,D)
    for j = 1:D, i = 1:D
        X[i,j] = exp(-β*h(i,j))
    end
    Xsq = sqrt(X)
    @tensor M1[a,b,c,d] := M[a',b',c',d']*Xsq[c',c]*Xsq[d',d]*Xsq[a,a']*Xsq[b,b']

    # For computing energy: M2 is a tensor across 2 nearest neighbor sites in the lattice, whose
    # expectation value in the converged fixed point of the transfer matrix represents the energy
    Y = zeros(D,D)
    for j = 1:D, i = 1:D
        Y[i,j] = h(i,j)*exp(-β*h(i,j))
    end
    @tensor M2[a,b1,b2,c,d2,d1] := M[a',b1',c1,d1']*Xsq[a,a']*Xsq[b1,b1']*Xsq[d1',d1]* Y[c1,c2]*
                                    M[c2,b2',c',d2']*Xsq[b2,b2']*Xsq[d2',d2]*Xsq[c',c]

    return M1, M2
end

classicalisingmpo(β; J = 1.0, h = 0.) = statmechmpo(β, (s1,s2)->-J*(-1)^(s1!=s2) - h/2*(s1==1 + s2==1), 2)

βc = log(1+sqrt(2))/2
β = 0.95*βc
M, M2 = classicalisingmpo(β)
D = 50
A = randn(D, 2, D) + im*randn(D, 2, D)
λ, AL, C, AR, FL, FR = vumps(A, M; tol = 1e-10)


# Compute energy:
#----------------
# Strategy 1:
# Compute energy by contracting M2 with two mps tensors in ket and bra, and the boundaries FL and FR.
# Make sure everything is normalized by dividing through the proper contribution of the partition function
@tensor AAC[α,s1,s2,β] := AL[α,s1,α']*C[α',β']*AR[β',s2,β]

@tensor Z2 = scalar(FL[α,c,β]*AAC[β,s1,s2,β']*M[c,t1,d,s1]*M[d,t2,c',s2]*FR[β',c',α']*conj(AAC[α,t1,t2,α']))
@tensor energy = scalar(FL[α,c,β]*AAC[β,s1,s2,β']*M2[c,t1,t2,c',s2,s1]*FR[β',c',α']*conj(AAC[α,t1,t2,α']) / Z2)

# Strategy 2:
# Compute energy using thermodynamic relations: Z = λ^N, i.e. λ is the partition function per site
# E = - d log(Z) / d β => energy (density) = - d log(λ) / d β
# where derivatives are evaluated using finite differences
dβ = 1.e-5
β′ = β + dβ
M′, = classicalisingmpo(β′)
λ′, = vumps(AL, M′; tol = 1e-10)
energy2 = -(log(λ′)-log(λ))/(β′-β)

@assert isapprox(energy2, 2*energy; rtol = 10*dβ)
 # factor 2 for counting horizontal and vertical links

# also compute free energy and entropy
f = -log(λ)/β
S = -β*(f - energy2)

f′ = -log(λ′)/β′
Salt = β^2*(f′-f)/(β′-β)
