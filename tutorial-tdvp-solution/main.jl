using Revise
Revise.includet("mps.jl")
Revise.includet("models.jl")

#script
N = 100
A = randmps(N, 2, 50);
M = isingmpo(N; h=0.9);

E1, A, F = dmrg1sweep!(A, M; verbose = true);
E2, A, F = dmrg1sweep!(A, M, F; verbose = true);
E3, A, F = dmrg2sweep!(A, M, F; verbose = true, truncdim = 200, truncerr = 1e-9);

using UnicodePlots

t = 0.
A0 = map(complex, A);
n = div(N,2);
sx = [0. 1.; 1. 0.];
sz = [1. 0.; 0. -1.];
for i = n-2:n+2
    @tensor A0[i][a,s,b] := sx[s,s']*A0[i][a,s',b];
end
p = scatterplot(real(measure1siteoperator(A0, sx)), ylim=[-1,1], color=:red, title="sx and sz at t = $t", width=80);
p = scatterplot!(p, real(measure1siteoperator(A0, sz)), color=:blue);
show(p)
show(scatterplot(entanglemententropy(A0), title="entanglement at t = $t", width=80))

dt = 0.01
t = dt
A0, F = tdvp1sweep!(dt, A0, M; verbose = false);
p = scatterplot(real(measure1siteoperator(A0, sx)), ylim=[-1,1], color=:red, title="sx and sz at t = $t", width=80);
p = scatterplot!(p, real(measure1siteoperator(A0, sz)), color=:blue);
show(p)
show(scatterplot(entanglemententropy(A0), title="entanglement at t = $t", width=80))

while t < .1
    global A0, F, t, dt
    A0, F = tdvp1sweep!(dt, A0, M; verbose = false);
    p = scatterplot(real(measure1siteoperator(A0, sx)), ylim=[-1,1], color=:red, title="sx and sz at t = $t", width=80);
    p = scatterplot!(p, real(measure1siteoperator(A0, sz)), color=:blue)
    show(p)
    show(scatterplot(entanglemententropy(A0), title="entanglement at t = $t", width=80))
    t += dt
end
