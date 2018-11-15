function xyzmpo(N::Int; Jx = 1.0, Jy=Jx, Jz=Jx, hx = 0., hz=0.)
    sx = [0. 1.; 1. 0.]
    isy = [0. 1.; -1. 0.]
    sz = [1. 0.; 0. -1.]
    u = [1. 0.; 0. 1.]

    D = 5 - iszero(Jx) - iszero(Jy) - iszero(Jz)
    M = zeros(D, 2, D, 2)
    M[1,:,1,:] = M[D,:,D,:] = u
    M[1,:,D,:] = -hx*sz + -hz*sz
    i = 2
    if !iszero(Jx)
        M[1,:,i,:] = -Jx*sx
        M[i,:,D,:] = sx
        i += 1
    end
    if !iszero(Jy)
        M[1,:,i,:] = Jy*isy
        M[i,:,D,:] = isy
        i += 1
    end
    if !iszero(Jz)
        M[1,:,i,:] = -Jz*sz
        M[i,:,D,:] = sz
        i += 1
    end
    return Any[M[1:1,:,:,:], fill(M, N-2)..., M[:,:,D:D, :]]
end

isingmpo(N::Int; J=1.0, h=1.0) = xyzmpo(N; Jx=J, Jy=0., Jz=0., hz=h, hx=0.)
heisenbergmpo(N::Int, J=1.0) = xyzmpo(N; Jx=J)
xxzmpo(N::Int, Δ = 1.0, J=1.0) = xyzmpo(N; Jx=J, Jy=J, Jz=J*Δ)
