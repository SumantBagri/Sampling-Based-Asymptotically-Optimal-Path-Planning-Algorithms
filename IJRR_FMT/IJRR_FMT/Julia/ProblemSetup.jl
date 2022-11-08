using Iterators

abstract ProblemSetup

immutable GeometricProblem <: ProblemSetup
    init::Vector{Float64}
    goal::Goal
    obs::ObstacleSet
    V0::Vector{Vector{Float64}}
    SS::BoundedEuclideanStateSpace
end

function GetValidGoalState(P::ProblemSetup)
    v = PointToState(SampleGoal(P.goal), P.SS)
    while ~PtValidQ(v, P.obs) || ~StateValidQ(v, P.SS)
        v = PointToState(SampleGoal(P.goal), P.SS)
    end
    return v
end

function SampleFree(P::ProblemSetup, N::Int64, ensure_goal::Bool = true, goal_bias::Float64 = 0.0)
    V = fill(P.init, N)
    sample_count = 1
    for v in chain(P.V0, iterate(x -> SampleSpace(P.SS), SampleSpace(P.SS))) # use of iterate() here is kinda dumb
        if PtValidQ(v, P.obs)
            sample_count = sample_count + 1
            if 0.0 < goal_bias && rand() < goal_bias
                V[sample_count] = GetValidGoalState(P)
            else
                V[sample_count] = v
            end
            if sample_count == N
                break
            end
        end
    end
    if ensure_goal && ~any(GoalPtQ(V, P.goal))
        V[N] = GetValidGoalState(P)
        # println("Explicit Goal Sample: ", V[N])
    end
    return V
end

function PlotProblemSetup(P::ProblemSetup)
    PlotBounds(P.SS.lo, P.SS.hi)
    PlotObstacles(P.obs)
    PlotGoal(P.goal)
end

# function PlotProblemSetup(P::ReedsSheppProblem)
#     PlotBounds(P.SS.lo, P.SS.hi)
#     PlotObstacles(P.obs)
#     PlotGoal(P.goal)
# end