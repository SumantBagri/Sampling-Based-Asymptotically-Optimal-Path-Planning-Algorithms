abstract Goal

immutable RectangleGoal <: Goal
    bounds::Array{Float64,2}
end

immutable BallGoal <: Goal
    center::Array{Float64,1}
    radius::Float64
end

immutable PointGoal <: Goal
    pt::Array{Float64,1}
end

function PlotGoal(G::RectangleGoal)
    PlotRect(G.bounds, "green")
end

function PlotGoal(G::BallGoal)
    PlotCircle(G.center, G.radius, xmax=1, ymax=1, color="green")
end

function PlotGoal(G::PointGoal)
    scatter(G.pt[1], G.pt[2], color="green", zorder=5)
end

## Rectangle Goal

function GoalPtQ(v::Array{Float64,1}, G::RectangleGoal)
    return all(G.bounds[:,1] .<= v .<= G.bounds[:,2])
end

function GoalPtQ(V::Array{Float64,2}, G::RectangleGoal)
    return all(G.bounds[:,1] .<= V .<= G.bounds[:,2], 1)
end

function SampleGoal(G::RectangleGoal)
    return G.bounds[:,1] + (G.bounds[:,2] - G.bounds[:,1]).*rand(size(G.bounds,1))
end

## Ball Goal

function GoalPtQ(v::Array{Float64,1}, G::BallGoal)
    return norm(v - G.center) <= G.radius
end

function GoalPtQ(V::Array{Float64,2}, G::BallGoal)
    return sum((V .- G.center).^2, 1) .<= G.radius^2
end

function SampleGoal(G::BallGoal)
    while true
        v = G.center + 2*G.radius*(rand(length(G.center)) - .5)
        if GoalPtQ(v, G)
            return v
        end
    end
end

## Point Goal

function GoalPtQ(v::Array{Float64,1}, G::PointGoal)
    return (v == G.pt)
end

function GoalPtQ(V::Array{Float64,2}, G::PointGoal)
    return all(V .== G.pt, 1)
end

function SampleGoal(G::PointGoal)
    return G.pt
end