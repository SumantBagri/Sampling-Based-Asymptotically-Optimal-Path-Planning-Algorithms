using Distance
using NearestNeighbors
using Devectorize

immutable BESSCuboidUnionRobot <: StateSpace
    dim::Int64
    lo::Vector{Float64}
    hi::Vector{Float64}
    centers::Array{Float64}
    halfwidths::Array{Float64}
    expanded_obs::Vector{AABoxes}   # bad, need non-immutable type or just refactor the code
    efflo::Array{Float64}
    effhi::Array{Float64}
end

function Volume(SS::BESSCuboidUnionRobot)
    return prod(SS.hi-SS.lo)
end

function FreeVolume(obs::ObstacleSet, SS::BESSCuboidUnionRobot)
    return Volume(SS) - Volume(obs)
end

function BESSCuboidUnionRobotUH(d::Int64, centers::Array{Float64}, halfwidths::Array{Float64}, obs::AABoxes)
    eos = AABoxes[]
    for i in 1:size(centers,2)
        eo = obs.boxes .+ [-halfwidths[:,i] halfwidths[:,i]]
        push!(eos, AABoxes(eo))
    end
    lo = zeros(d)
    hi = ones(d)
    return BESSCuboidUnionRobot(d, lo, hi, centers, halfwidths, eos, lo - minimum(centers-halfwidths, 2), hi - maximum(centers+halfwidths, 2))
end

function PairwiseDistances(V::Array{Float64, 2}, SS::BESSCuboidUnionRobot, r_bound::Float64)
    return pairwise(Euclidean(), V)
end

function PairwiseDistances(V::Vector{Vector{Float64}}, SS::BESSCuboidUnionRobot, r_bound::Float64)
    return pairwise(Euclidean(), hcat(V...))
end

function Steer(v::Vector{Float64}, w::Vector{Float64}, eps::Float64, normvw = norm(w - v))
    return v + (w - v) * min(eps/normvw, 1)
end

function SampleSpace(SS::BESSCuboidUnionRobot)
    return SS.lo + rand(SS.dim).*(SS.hi-SS.lo)
end

function PointToState(v::Vector{Float64}, SS::BESSCuboidUnionRobot)   # for explicit goal sampling
    return v
end

function MotionValidQ(v::Array{Float64}, w::Array{Float64}, obs::ObstacleSet, SS::BESSCuboidUnionRobot)
    for i in 1:size(SS.centers, 2) 
        if !MotionValidQ(v .+ SS.centers[:,i], w .+ SS.centers[:,i], SS.expanded_obs[i])
            return false
        end
    end
    return true
end

function GoalPtQ(V::Vector{Vector{Float64}}, G::Goal)
    return GoalPtQ(hcat(V...), G)
end

function PtValidQ(v::AbstractVector, obs::ObstacleSet, SS::BESSCuboidUnionRobot)
    if !StateValidQ(v, SS)
        return false
    end
    for i in 1:size(SS.centers, 2)
        if !PtValidQ(v .+ SS.centers[:,i], SS.expanded_obs[i])
            return false
        end
    end
    return true
end

function StateValidQ(v::AbstractVector, SS::BESSCuboidUnionRobot)
    # return all(SS.lo .<= (v .+ SS.cph) .<= SS.hi) && all(SS.lo .<= (v .+ SS.cmh) .<= SS.hi)
    # return all(SS.efflo .<= v .<= SS.effhi)
    for i in 1:length(v)
        if !(SS.efflo[i] <= v[i] <= SS.effhi[i])
            return false
        end
    end
    return true
end

### SHORTCUTTING (CHEATS)

function ShortcutPath(path::Array{Float64,2}, obs::Array{Float64,3})
    N = size(path, 2)
    if N == 2
        return path
    end
    if MotionValidQ(path[:,1], path[:,end], obs)
        return [path[:,1] path[:,end]]
    end
    mid = iceil(N/2)
    return [ShortcutPath(path[:,1:mid], obs)[:,1:end-1] ShortcutPath(path[:,mid:end], obs)]
end

function CutCorner(V::Array{Float64,2}, obs::Array{Float64,3})
    m1 = (V[:,1] + V[:,2])/2
    m2 = (V[:,3] + V[:,2])/2
    while ~MotionValidQ(m1, m2, obs)
        m1 = (m1 + V[:,2])/2
        m2 = (m2 + V[:,2])/2
    end
    return [V[:,1] m1 m2 V[:,3]]
end

function AdaptiveShortcutPath(path::Array{Float64,2}, obs::Array{Float64,3}, iterations::Int64)
    while (short_path = ShortcutPath(path, obs)) != path
        path = short_path
    end
    for i in 1:iterations
        path = [path[:,1] hcat([CutCorner(path[:,j-1:j+1], obs)[:, 2:3] for j in 2:size(path,2)-1]...) path[:,end]]
        while (short_path = ShortcutPath(path, obs)) != path
            path = short_path
        end
    end
    return path
end

### NEAREST NEIGHBOR STUFF

abstract EuclideanNN

immutable EuclideanNNBrute <: EuclideanNN
    D::Matrix{Float64}
    cache::Vector{Vector{Int64}}
    kNNr::Vector{Float64}
end

immutable EuclideanNNKDTree <: EuclideanNN
    V::Vector{Vector{Float64}}
    KDT::KDTree
    cache::Vector{Any}
    kNNr::Vector{Float64}
end

## Brute

function EuclideanNNBrute(V::Vector{Vector{Float64}})
    return EuclideanNNBrute(pairwise(Euclidean(), hcat(V...)), fill(Int64[], length(V)), zeros(length(V)))
end

function NearestR(NN::EuclideanNNBrute, v::Int64, r::Float64, usecache::Bool = true)
    if !usecache || isempty(NN.cache[v])
        nn_bool = NN.D[:,v] .< r
        nn_bool[v] = false
        nn_idx = find(nn_bool)
        if usecache
            NN.cache[v] = nn_idx
        end
    else
        nn_idx = NN.cache[v]
    end
    return nn_idx, NN.D[nn_idx, v]
end

function NearestK(NN::EuclideanNNBrute, v::Int64, k::Int64, usecache::Bool = true)
    if !usecache || isempty(NN.cache[v])
        r = select!(NN.D[:,v], k+1)
        nn_bool = NN.D[:,v] .<= r
        nn_bool[v] = false
        nn_idx = find(nn_bool)
        if usecache
            NN.cache[v] = nn_idx
            NN.kNNr[v] = r
        end
    else
        nn_idx = NN.cache[v]
    end
    return nn_idx, NN.D[nn_idx, v]
end

## KDTree

function EuclideanNNKDTree(V::Vector{Vector{Float64}})
    return EuclideanNNKDTree(V, KDTree(hcat(V...), Euclidean()), {() for i in 1:length(V)}, zeros(length(V)))
end

function NearestK(NN::EuclideanNNKDTree, v::Int64, k::Int64, usecache::Bool = true) # not bothering with usecache
    if isempty(NN.cache[v])
        nn_idx, nn_D = nearest(NN.KDT, NN.V[v], k+1)
        x = findin(nn_idx, v)
        deleteat!(nn_idx, x)
        deleteat!(nn_D, x)
        NN.cache[v] = (nn_idx, nn_D)
        NN.kNNr[v] = maximum(nn_D)
    end
    return NN.cache[v]
end

## Both

function NearestR(NN::EuclideanNN, v::Int64, r::Float64, filter::BitVector, usecache::Bool = true)
    nn_idx, D = NearestR(NN, v, r, usecache)
    return nn_idx[filter[nn_idx]], D[filter[nn_idx]]
end

function NearestK(NN::EuclideanNN, v::Int64, k::Int64, filter::BitVector, usecache::Bool = true)
    nn_idx, D = NearestK(NN, v, k, usecache)
    return nn_idx[filter[nn_idx]], D[filter[nn_idx]]
end

function MutualNearestK(NN::EuclideanNN, v::Int64, k::Int64, filter::BitVector)
    nn_idx, D = NearestK(NN, v, k, true)
    for w in nn_idx
        NearestK(NN, w, k, true)       # should refactor to eliminate trashed returns :/ shouldn't matter much
    end
    # println(NN.kNNr[nn_idx])
    meta_idx = filter[nn_idx] & (D .<= NN.kNNr[nn_idx])      # also, name vars better
    # println(nn_idx[meta_idx], D[meta_idx])
    return nn_idx[meta_idx], D[meta_idx]
end

##

function NearestPoint(w::Vector{Float64}, V::Vector{Vector{Float64}}, SS::BESSCuboidUnionRobot)
    return findmin(colwise(Euclidean(), hcat(V...), w))
end

function NearestPointsR(w::Vector{Float64}, V::Vector{Vector{Float64}}, r::Float64, SS::BESSCuboidUnionRobot)
    D = colwise(Euclidean(), hcat(V...), w)
    return D[D .< r], find(D .< r)
end