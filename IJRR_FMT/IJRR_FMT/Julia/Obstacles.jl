using ArrayViews
using Devectorize

abstract ObstacleSet

immutable ObstacleList <: ObstacleSet
    list::Vector{ObstacleSet}
end

immutable AABoxes <: ObstacleSet
    boxes::Array{Float64,3}
end

### ObstacleList

function PtValidQ(v::Array{Float64}, obs_list::ObstacleList)
    return all([PtValidQ(v, obs) for obs in obs_list.list])
end

function MotionValidQ(v::Array{Float64}, w::Array{Float64}, obs_list::ObstacleList)
    return all([MotionValidQ(v, w, obs) for obs in obs_list.list])
end

function PathValidQ(path::Array{Float64,2}, obs_list::ObstacleList)
    return all([PathValidQ(path, obs) for obs in obs_list.list])
end

function ClosestObsPt(v::AbstractVector, obs_list::ObstacleList)
    mapreduce(obs -> ClosestObsPt(v, obs), (x,y) -> (x[2] < y[2] ? x : y), obs_list.list)
end

### AABoxes

function PtValidQ(v::Array{Float64}, obs::AABoxes)
    return PtValidQ(v, obs.boxes)
end

function MotionValidQ(v::Array{Float64}, w::Array{Float64}, obs::AABoxes)
    return MotionValidQ(v, w, obs.boxes)
end

function PathValidQ(path::Array{Float64,2}, obs::AABoxes)
    return PathValidQ(path, obs.boxes)
end

function ClosestObsPt(v::AbstractVector, obs::AABoxes)
    return ClosestObsPt(v, obs.boxes)
end

## Volume

function Volume(obs::ObstacleSet)
    return NaiveVolume(obs)
end

function NaiveVolume(obs_list::ObstacleList)
    return sum([NaiveVolume(obs) for obs in obs_list.list])
end

function NaiveVolume(obs::AABoxes)
    return sum(mapslices(prod, obs.boxes[:,2,:] - obs.boxes[:,1,:], [1,2]))
end

### PLOTTING

function PlotObstacles(obs_list::ObstacleList)
    [PlotObstacles(obs) for obs in obs_list.list]
end

function PlotObstacles(obs::AABoxes)
    mapslices(o -> PlotRect(o, "red"), obs.boxes, [1,2])
end

### REAL STUFF

# function PtValidQ(v::AbstractVector, obs::Array{Float64,3})
#     return ~any(all(obs[:,1,:] .<= v .<= obs[:,2,:], [1,2]))
# end

function PtValidQ(v::AbstractVector, obs::AbstractMatrix)
    for i = 1:size(obs,1)
        if !(obs[i,1] <= v[i] <= obs[i,2])
            return false
        end
    end
    return true
end

function PtValidQ(v::AbstractVector, obs::Array{Float64,3})
    for k = 1:size(obs,3)
        PtValidQ(v, view(obs,:,:,k)) && return false
    end
    return true
end

# function MotionValidQ(v::AbstractVector, w::AbstractVector, obs::Array{Float64,3})
#     bp = vec(all((obs[:,2,:] .>= min(v,w)) & (obs[:,1,:] .<= max(v,w)), 1))
#     return ~any(bp) || MotionValidQ(v, w, obs, bp)
# end

# function MotionValidQ(v::AbstractVector, w::AbstractMatrix, obs::Array{Float64,3})
#     bb_min = broadcast(min, v, w)
#     bb_max = broadcast(max, v, w)
#     bps = all((obs[:,2,:] .>= bb_min) & (obs[:,1,:] .<= bb_max), 1)
#     return [(~any(bps[:,i,:]) || MotionValidQ(v, w[:,i], obs, vec(bps[:,i,:]))) for i in 1:size(w, 2)]
# end

# function MotionValidQ(v::AbstractVector, w::AbstractVector, obs::Array{Float64,3}, bp::BitArray{1})
#     corners = (v .< obs[:,1,bp]).*obs[:,1,bp] + (v .>= obs[:,1,bp]).*obs[:,2,bp]
#     intersection_pts = v .+ mapslices(x -> (w .- v)*x', (corners .- v)./(w .- v), [1,2])
#     return ~any(all(broadcast(|, diagm(trues(length(v))), (obs[:,1,bp] .<= intersection_pts .<= obs[:,2,bp])), 1))
# end


function BroadphaseValidQ(bb_min::AbstractVector, bb_max::AbstractVector, obs::Array{Float64,3}, k::Integer)
    for i = 1:size(obs,1)
        if obs[i,2,k] < bb_min[i] || obs[i,1,k] > bb_max[i]
            return true
        end
    end
    return false
end

function MotionValidQ(v::AbstractVector, w::AbstractVector, obs::Array{Float64,3})
    bb_min = min(v,w)
    bb_max = max(v,w)
    for k = 1:size(obs,3)
        if !BroadphaseValidQ(bb_min, bb_max, obs, k)
            if !MotionValidQ(v, w, view(obs,:,:,k))
                return false
            end
        end
    end
    return true
end

function FaceContainsProjectionQ(v::AbstractVector, v_to_w::AbstractVector, lambda::Number, j::Integer, obs::AbstractMatrix)
    for i = 1:size(obs,1)
        if i != j && !(obs[i,1] <= v[i] + v_to_w[i]*lambda <= obs[i,2])
            return false
        end
    end
    return true
end

function MotionValidQ(v::AbstractVector, w::AbstractVector, obs::AbstractMatrix)
    v_to_w = w - v
    @devec lambdas = (blend(v .< obs[:,1], obs[:,1], obs[:,2]) .- v) ./ v_to_w
    for i in 1:size(obs,1)
        FaceContainsProjectionQ(v, v_to_w, lambdas[i], i, obs) && return false
    end
    return true
end

function PathValidQ(path::Array{Float64,2}, obs::Array{Float64,3})
    return NaivePathValidQ(path, obs)
end

function NaivePathValidQ(path::Array{Float64,2}, obs::Array{Float64,3})
    # return all([MotionValidQ(path[:,i], path[:,i+1], obs) for i in 1:size(path,2)-1])
    for i in 1:size(path,2)-1
        if !MotionValidQ(path[:,i], path[:,i+1], obs)
            return false
        end
    end
    return true
end

function BoxClamp(v::AbstractVector, lo::AbstractVector, hi::AbstractVector)
    [clamp() for i in 1:length(v)]
end

function ClosestObsPt(v::AbstractVector, obs::Array{Float64,3})
    CPs = reduce((x,y) -> (x[2] < y[2] ? x : y), [map(clamp, v, obs[:,1,i], obs[:,2,i]) |> x -> (x, norm(x-v)) for i in 1:size(obs,3)]) # rewrite as mapreduce with COP(v, Matrix)
end