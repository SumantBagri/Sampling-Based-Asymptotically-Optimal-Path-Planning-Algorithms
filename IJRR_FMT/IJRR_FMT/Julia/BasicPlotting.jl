using PyPlot

function PlotRect(r, col="black"; kwargs...)
    plt.fill(r[1,:][[1,2,2,1,1]], r[2,:][[1,1,2,2,1]], color=col, edgecolor="black", zorder=0; kwargs...)
end

function PlotCircle(c, r; xmax = Inf, ymax = Inf, kwargs...)
    plt.fill(min(c[1] + r*cos(linspace(0, 2pi, 40)), xmax), min(c[2] + r*sin(linspace(0, 2pi, 40)), ymax), edgecolor="black", zorder=0; kwargs...)
end

function PlotEllipse(c::Vector{Float64}, a::Float64, b::Float64, t::Float64; xmax = Inf, ymax = Inf, kwargs...)
    XY = [a*cos(linspace(0, 2pi, 40)) b*sin(linspace(0, 2pi, 40))]*[cos(t) sin(t); -sin(t) cos(t)]
    plt.fill(min(c[1] + XY[:,1], xmax), min(c[2] + XY[:,2], ymax), edgecolor="black", zorder=0; kwargs...)
end

function PlotEllipse(c::Vector{Float64}, Sigma::Array{Float64,2}; xmax = Inf, ymax = Inf, kwargs...)
    XY = [cos(linspace(0, 2pi, 40)) sin(linspace(0, 2pi, 40))]*chol(Sigma)
    plt.fill(min(c[1] + XY[:,1], xmax), min(c[2] + XY[:,2], ymax), edgecolor="black", zorder=0; kwargs...)
end

function PlotBounds(lo::Vector{Float64} = zeros(2), hi::Vector{Float64} = ones(2))
    plt.plot([lo[1],hi[1],hi[1],lo[1],lo[1]], [lo[2],lo[2],hi[2],hi[2],lo[2]], color="black", linewidth=1.0, linestyle="-")
    axis("equal")
end

function PlotGraph(V::Array{Float64,2}, F, col="black"; kwargs...)  # learn how to just pass kwargs
    scatter(V[1,:], V[2,:], zorder=1; kwargs...)
    X = vcat([V[1,idx_list] for idx_list in findn(triu(F))]..., fill(nothing, 1, sum(triu(F))))[:]
    Y = vcat([V[2,idx_list] for idx_list in findn(triu(F))]..., fill(nothing, 1, sum(triu(F))))[:]
    plt.plot(X, Y, color=col, linewidth=.5, linestyle="-", zorder=1; kwargs...)
end

function PlotGraph(V::Vector{Vector{Float64}}, F, col="black"; kwargs...)
    PlotGraph(hcat(V...), F, col; kwargs...)
end

function PlotTree(V::Array{Float64,2}, A, col="black"; kwargs...)
    scatter(V[1,:], V[2,:], zorder=1; kwargs...)
    X = vcat(V[1,find(A)], V[1,A[find(A)]], fill(nothing, 1, countnz(A)))[:]
    Y = vcat(V[2,find(A)], V[2,A[find(A)]], fill(nothing, 1, countnz(A)))[:]
    plt.plot(X, Y, color=col, linewidth=.5, linestyle="-", zorder=1; kwargs...)
end

function PlotTree(V::Vector{Vector{Float64}}, A, col="black"; kwargs...)
    PlotTree(hcat(V...), A, col; kwargs...)
end

function PlotSolution(V::Array{Float64,2}, sol, col="black"; kwargs...)
    plt.plot(V[1,sol]', V[2, sol]', color=col, linewidth=2.0, linestyle="-", zorder=2; kwargs...)
end

function PlotSolution(V::Vector{Vector{Float64}}, sol, col="black"; kwargs...)
    PlotSolution(hcat(V...), sol, col; kwargs...)
end

function PlotPath(P, col="black"; kwargs...)
    plt.plot(P[1,:]', P[2,:]', color=col, linewidth=1.0, linestyle="-", zorder=2; kwargs...)
end