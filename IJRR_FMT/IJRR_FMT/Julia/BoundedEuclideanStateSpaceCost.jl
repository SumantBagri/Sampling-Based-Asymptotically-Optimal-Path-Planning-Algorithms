using Distance
using NearestNeighbors
using Devectorize
using ODE

immutable BoundedEuclideanStateSpaceCost <: StateSpace
    dim::Int64
    lo::Vector{Float64}
    hi::Vector{Float64}
    costf::Function
    c_min::Float64
end

function Volume(SS::BoundedEuclideanStateSpaceCost)
    return prod(SS.hi-SS.lo)
end

function FreeVolume(obs::ObstacleSet, SS::BoundedEuclideanStateSpaceCost)
    return Volume(SS) - Volume(obs)
end

# function UnitHypercube(d::Int64)
#     return BoundedEuclideanStateSpaceCost(d, zeros(d), ones(d))
# end

function TrapIntegrate(f::Function, v::Vector{Float64}, w::Vector{Float64}, N::Int64)
    s = 0
    s = s + f(v) + f(w)
    for x in [1:N-1]/N
        s = s + 2*f(x*v + (1-x)*w)
    end
    return s*norm(v-w) / (2*N)
end

function PairwiseDistances(V::Array{Float64, 2}, SS::BoundedEuclideanStateSpaceCost, r_bound::Float64)
    ED = pairwise(Euclidean(), V)
    N = size(V,2)
    D = fill(Inf, N, N)
    for j in 1:N
        for i in j+1:N
            if ED[i,j] < r_bound / SS.c_min
                @inbounds D[i,j] = TrapIntegrate(SS.costf, V[:,i], V[:,j], 10)
                @inbounds D[j,i] = D[i,j]
            end
        end
    end
    return D
end

function PairwiseDistances(V::Vector{Vector{Float64}}, SS::BoundedEuclideanStateSpaceCost, r_bound::Float64)
    return PairwiseDistances(hcat(V...), SS, r_bound)
end

# function Steer(v::Vector{Float64}, w::Vector{Float64}, eps::Float64, normvw = norm(w - v))
#     return v + (w - v) * min(eps/normvw, 1)
# end

function SampleSpace(SS::BoundedEuclideanStateSpaceCost)
    return SS.lo + rand(SS.dim).*(SS.hi-SS.lo)
end

function PointToState(v::Vector{Float64}, SS::BoundedEuclideanStateSpaceCost)   # for explicit goal sampling
    return v
end

function MotionValidQ(v::Array{Float64}, w::Array{Float64}, obs::ObstacleSet, SS::BoundedEuclideanStateSpaceCost)
    return MotionValidQ(v, w, obs)
end

function GoalPtQ(V::Vector{Vector{Float64}}, G::Goal)
    return GoalPtQ(hcat(V...), G)
end

function StateValidQ(v::AbstractVector, SS::BoundedEuclideanStateSpaceCost)
    return all(SS.lo .<= v .<= SS.hi)
end

# ### SHORTCUTTING (CHEATS)

# function ShortcutPath(path::Array{Float64,2}, obs::Array{Float64,3})
#     N = size(path, 2)
#     if N == 2
#         return path
#     end
#     if MotionValidQ(path[:,1], path[:,end], obs)
#         return [path[:,1] path[:,end]]
#     end
#     mid = iceil(N/2)
#     return [ShortcutPath(path[:,1:mid], obs)[:,1:end-1] ShortcutPath(path[:,mid:end], obs)]
# end

# function CutCorner(V::Array{Float64,2}, obs::Array{Float64,3})
#     m1 = (V[:,1] + V[:,2])/2
#     m2 = (V[:,3] + V[:,2])/2
#     while ~MotionValidQ(m1, m2, obs)
#         m1 = (m1 + V[:,2])/2
#         m2 = (m2 + V[:,2])/2
#     end
#     return [V[:,1] m1 m2 V[:,3]]
# end

# function AdaptiveShortcutPath(path::Array{Float64,2}, obs::Array{Float64,3}, iterations::Int64)
#     while (short_path = ShortcutPath(path, obs)) != path
#         path = short_path
#     end
#     for i in 1:iterations
#         path = [path[:,1] hcat([CutCorner(path[:,j-1:j+1], obs)[:, 2:3] for j in 2:size(path,2)-1]...) path[:,end]]
#         while (short_path = ShortcutPath(path, obs)) != path
#             path = short_path
#         end
#     end
#     return path
# end