function FMTRadius(N::Int64, obs::Array{Float64,3}, eta::Float64)
    dim = size(obs,1)
    mu = 1 - sum(mapslices(prod, obs[:,2,:] - obs[:,1,:], [1,2]))
    return FMTRadius(N, dim, mu, eta)
end

function FMTRadius(N::Int64, dim::Int64, mu::Float64, eta::Float64)
    #     r = eta*sqrt(2*mu/pi*log(N)/N) for RRTstar paper
    return eta*2*(1/dim*mu/(pi^(dim/2)/gamma(dim/2+1))*log(N)/N)^(1/dim)
end

function PRMstar(init::Array{Float64,1}, goal::Goal, obs::Array{Float64,3}, N::Int64, eta::Float64, V::Array{Float64, 2}=Array(Float64,0,0); r::Float64 = FMTRadius(N, obs, eta))
    V = [init SampleFree(obs, goal, N-1, V)]
    D = pairwise(Euclidean(), V)

    F = (D .< r) & ~diagm(trues(N))
    for i in 1:N
        F[F[:,i],i] = MotionValidQ(V[:,i], V[:,F[:,i]], obs)
    end

    V_goal = find(GoalPtQ(V, goal))
    G = simple_inclist(N)
    for i in find(F)
        add_edge!(G, ind2sub(size(F), i)...)
    end
    sp = dijkstra_shortest_paths(G, D[find(F)], 1)
    c, idx = findmin(sp.dists[V_goal])
    if c == Inf
        return Inf, Int64[], V, F
    end
    sol = [V_goal[idx]]
    while sol[end] != 1
        push!(sol, sp.parents[sol[end]])
    end
    return c, sol, V, F
end

function FMTstar(init::Array{Float64,1}, goal::Goal, obs::Array{Float64,3}, N::Int64, eta::Float64, V::Array{Float64, 2}=Array(Float64,0,0); r::Float64 = FMTRadius(N, obs, eta))
    V = [init SampleFree(obs, goal, N-1, V)]
    D = pairwise(Euclidean(), V)

    NN = (D .< r) & ~diagm(trues(N))  # consider switching to array indexing
    A = zeros(Int64,N)
    W = trues(N)
    W[1] = false;
    H = falses(N)
    H[1] = true;
    C = zeros(N)
    z = 1

    while ~GoalPtQ(V[:,z], goal)
        H_new = Int64[]
        for x in find(NN[:,z] & W)
            Y_near = find(NN[:,x] & H)
            c_min, y_idx = findmin(C[Y_near] + D[Y_near,x])
            y_min = Y_near[y_idx]
            if MotionValidQ(V[:,x], V[:,y_min], obs)
                A[x] = y_min
                C[x] = c_min
                push!(H_new, x)
                W[x] = false
            end
        end
        H[H_new] = true
        H[z] = false
        if any(H)
            z = find(H)[indmin(C[H])]
        else
            return Inf, Int64[], V, A
        end
    end

    sol = [z]
    while sol[end] != 1
        push!(sol, A[sol[end]])
    end
    return C[z], sol, V, A
end