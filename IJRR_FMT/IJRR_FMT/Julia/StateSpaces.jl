abstract StateSpace

### DEFAULTS

function PtValidQ(v, obs, SS::StateSpace)
    return PtValidQ(v, obs)
end

function MotionValidQ(v, w, obs, SS::StateSpace)
    return MotionValidQ(v, w, obs)
end

function PathValidQ(path, obs, SS::StateSpace)
    return PathValidQ(path, obs)
end

include("BoundedEuclideanStateSpace.jl")
# include("ReedsSheppStateSpace.jl")