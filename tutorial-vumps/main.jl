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
    @tensor M[a,b,c,d] := M[a,b,c',d']*X[c',c]*X[d',d]
    return M
end

classicalisingmpo(β; J = 1.0, h = 0.) = statmechmpo(β, (s1,s2)->J*(-1)^(s1==s2) - h/2*(s1==1 + s2==1), 2)

βc = log(1+sqrt(2))/2
M = classicalisingmpo(0.1)

D = 50
A = randn(D, 2, D)
vumps(A, M)
