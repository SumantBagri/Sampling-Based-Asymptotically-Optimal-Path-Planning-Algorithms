using Graphs

function FMTRadius(N::Int64, obs::Array{Float64,3}, rm::Float64)
    dim = size(obs,1)
    mu = 1 - sum(mapslices(prod, obs[:,2,:] - obs[:,1,:], [1,2]))
    return FMTRadius(N, dim, mu, rm)
end

function FMTRadius(N::Int64, dim::Int64, mu::Float64, rm::Float64)
    #     r = rm*sqrt(2*mu/pi*log(N)/N) for RRTstar paper
    return rm*2*(1/dim*mu/(pi^(dim/2)/gamma(dim/2+1))*log(N)/N)^(1/dim)
end

function FMTRadius(N::Int64, P::GeometricProblem, rm::Float64)
    return FMTRadius(N, P.SS.dim, FreeVolume(P.obs, P.SS), rm)
end

function FMTstar(P::ProblemSetup, N::Int64, rm::Float64; r::Float64 = FMTRadius(N, P, rm), kNN::Bool = false, kdt::Bool = true)
    tic()
    collision_checks = 0

    V = SampleFree(P, N)
    if kNN && kdt
        NN = EuclideanNNKDTree(V)
    else
        NN = EuclideanNNBrute(V)
    end

    A = zeros(Int64,N)
    W = trues(N)
    W[1] = false
    H = falses(N)
    H[1] = true
    C = zeros(N)
    z = 1

    k = min(iceil((2*rm)^P.SS.dim*(e/P.SS.dim)*log(N)), N-1)

    while ~GoalPtQ(V[z], P.goal)      # maybe V should always be an Array{Float64,2} instead of a Vector{StateType}
        H_new = Int64[]
        for x in (kNN ? MutualNearestK(NN, z, k, W)[1] : NearestR(NN, z, r, W)[1])
            Y_near, D_near = (kNN ? NearestK(NN, x, k, H) : NearestR(NN, x, r, H))
            c_min, y_idx = findmin(C[Y_near] + D_near)
            y_min = Y_near[y_idx]
            if (collision_checks = collision_checks + 1; MotionValidQ(V[x], V[y_min], P.obs, P.SS))
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
            break
            # return Inf, Int64[], V, A
        end
    end

    sol = [z]
    while sol[1] != 1
        unshift!(sol, A[sol[1]])
    end

    solution_metadata = {
        "radius_multiplier" => rm,
        "collision_checks" => collision_checks,
        "elapsed" => toq(),     # perhaps should move outside, but practically no difference as 1 function call is 0s
        "num_samples" => N,
        "cost" => C[z],
        # "test_config" => "boxes",
        "planner" => "FMT",
        "solved" => GoalPtQ(V[z], P.goal)
    }

    return solution_metadata, sol, V, A
end

function PRMstar(P::ProblemSetup, N::Int64, rm::Float64; r::Float64 = FMTRadius(N, P, rm), kNN::Bool = false, kdt::Bool = true)
    tic()
    collision_checks = 0

    V = SampleFree(P, N)
    if kNN && kdt
        D = PairwiseDistances(V, P.SS, r)

        F = (D .< r) & ~diagm(trues(N))
    else
        NN = EuclideanNNKDTree(V)
        D = zeros(N,N)

    end

    for j = 1:N
        for i = j+1:N
            @inbounds F[i,j] = F[i,j] && (collision_checks = collision_checks + 1; MotionValidQ(V[i], V[j], P.obs, P.SS))
            @inbounds F[j,i] = F[i,j]
        end
    end

    V_goal = find(GoalPtQ(V, goal))
    G = simple_inclist(N)
    for i in find(F)
        add_edge!(G, ind2sub(size(F), i)...)
    end
    sp = dijkstra_shortest_paths(G, D[find(F)], 1)
    c, idx = findmin(sp.dists[V_goal])
    if c == Inf
        sol = Int64[]
    else
        sol = [V_goal[idx]]
        while sol[1] != 1
            unshift!(sol, sp.parents[sol[1]])
        end
    end

    solution_metadata = {
        "radius_multiplier" => rm,
        "collision_checks" => collision_checks,
        "elapsed" => toq(),     # perhaps should move outside, but practically no difference as 1 function call is 0s
        "num_samples" => N,
        "cost" => c,
        # "test_config" => "boxes",
        "planner" => "PRMstar",
        "solved" => !(c==Inf)
    }

    return solution_metadata, sol, V, F
end

# NEEDS THE solution_metadata TREATMENT AFTER THE JOURNAL SIMS ARE DONE
# function RRT(P::ProblemSetup, N_max::Int64, steer_eps = .1)
#     V = fill(P.init, N_max)
#     R = SampleFree(P, N_max, false)
#     A = zeros(Int64, N_max)
#     N = 1

#     for i = 2:N_max
#         d_nrst, k_nrst = NearestPoint(R[i], V[1:N], P.SS)
#         v_new = Steer(V[k_nrst], R[i], steer_eps, d_nrst)
#         if MotionValidQ(V[k_nrst], v_new, P.obs, P.SS)
#             N = N + 1
#             V[N] = v_new
#             A[N] = k_nrst
#             if GoalPtQ(v_new, P.goal)
#                 break
#             end
#         end
#     end

#     sol = [N]
#     while sol[1] != 1
#         unshift!(sol, A[sol[1]])
#     end
#     c = sum([norm(x) for x in diff(V[sol])])

#     return c, sol, V[1:N], A[1:N]
# end

function RRTstarRadius(N::Int64, dim::Int64, mu::Float64, rm::Float64)
    #     r = rm*sqrt(2*mu/pi*log(N)/N) for RRTstar paper
    return rm*2*(1/dim*mu/(pi^(dim/2)/gamma(dim/2+1))*log(N)/N)^(1/dim)
end

function RRTstar(P::ProblemSetup, N_max::Int64, steer_eps = .1, rm::Float64 = 1.1, goal_bias = 0.05; mu::Float64 = FreeVolume(P.obs, P.SS))
    tic()
    collision_checks = 0

    V = fill(P.init, N_max)
    R = SampleFree(P, N_max, false, goal_bias)
    A = zeros(Int64, N_max)
    C = zeros(N_max)
    N = 1

    for i = 2:N_max
        d_nrst, k_nrst = NearestPoint(R[i], V[1:N], P.SS)
        v_new = Steer(V[k_nrst], R[i], steer_eps, d_nrst)
        if (collision_checks = collision_checks + 1; MotionValidQ(V[k_nrst], v_new, P.obs, P.SS))
            d_near, k_near = NearestPointsR(v_new, V[1:N], RRTstarRadius(N, P.SS.dim, mu, rm), P.SS)
            N = N + 1
            V[N] = v_new
            k_min = k_nrst
            c_min = C[k_nrst] + norm(v_new - V[k_nrst])     # yeah, geometric-specific, deal with it for now
            c_near = d_near + C[k_near]
            for j = sortperm(c_near)
                if c_near[j] >= c_min
                    break
                end
                if (collision_checks = collision_checks + 1; MotionValidQ(V[k_near[j]], v_new, P.obs, P.SS))
                    k_min = k_near[j]
                    c_min = c_near[j]
                end
            end
            A[N] = k_min
            C[N] = c_min

            k_rw = find(C[N] + d_near .< C[k_near])
            for j = k_rw
                if (collision_checks = collision_checks + 1; MotionValidQ(v_new, V[k_near[j]], P.obs, P.SS))
                    A[k_near[j]] = N
                    c_delta = C[N] + d_near[j] - C[k_near[j]]
                    progeny = k_near[j]
                    while ~isempty(progeny)
                        C[progeny] = C[progeny] + c_delta
                        progeny = findin(A, progeny)
                    end
                end
            end
        end
    end

    V_goal = find(GoalPtQ(V[1:N], goal))
    if isempty(V_goal)
        c = Inf
        sol = Int64[]
    else
        c, k = findmin(C[V_goal])
        sol = [V_goal[k]]
        while sol[1] != 1
            unshift!(sol, A[sol[1]])
        end
    end

    solution_metadata = {
        "radius_multiplier" => rm,
        "collision_checks" => collision_checks,
        "elapsed" => toq(),     # perhaps should move outside, but practically no difference as 1 function call is 0s
        "num_samples" => N_max,
        "cost" => c,
        # "test_config" => "boxes",
        "planner" => "RRTstar",
        "solved" => !(c==Inf)
    }

    return solution_metadata, sol, V[1:N], A[1:N]
end








#### OLD SCHOOL SHIT

function FMTstarOG(P::ProblemSetup, N::Int64, eta::Float64; r::Float64 = FMTRadius(N, P, eta), kNN::Bool = true)
    tic()
    collision_checks = 0

    V = SampleFree(P, N)
    D = PairwiseDistances(V, P.SS, r)

    NN = (D .< r) & ~diagm(trues(N))  # consider switching to array indexing
    A = zeros(Int64,N)
    W = trues(N)
    W[1] = false
    H = falses(N)
    H[1] = true
    C = zeros(N)
    z = 1

    while ~GoalPtQ(V[z], P.goal)      # maybe V should always be an Array{Float64,2} instead of a Vector{StateType}
        H_new = Int64[]
        for x in find(NN[:,z] & W)
            Y_near = find(NN[:,x] & H)
            c_min, y_idx = findmin(C[Y_near] + D[Y_near,x])
            y_min = Y_near[y_idx]
            if (collision_checks = collision_checks + 1; MotionValidQ(V[x], V[y_min], P.obs, P.SS))
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
            break
            # return Inf, Int64[], V, A
        end
    end

    sol = [z]
    while sol[1] != 1
        unshift!(sol, A[sol[1]])
    end

    solution_metadata = {
        "radius_multiplier" => eta,
        "collision_checks" => collision_checks,
        "elapsed" => toq(),     # perhaps should move outside, but practically no difference as 1 function call is 0s
        "num_samples" => N,
        "cost" => C[z],
        # "test_config" => "boxes",
        "planner" => "FMT",
        "solved" => GoalPtQ(V[z], P.goal)
    }

    return solution_metadata, sol, V, A
end