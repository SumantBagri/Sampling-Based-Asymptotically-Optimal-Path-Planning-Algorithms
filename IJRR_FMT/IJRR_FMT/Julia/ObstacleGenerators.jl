# function SimpleMaze(d)
#   fill_box(bx, d) = [bx[0]+[0]*d, bx[1]+[1]*d]
# end

# function barriers(d)
#   if d == 1
#       return []
#   end
#   dboxes = zeros()
# end

function RandomNonIntersectingBoxes(d, coverage_fraction)
    return RandomNonIntersectingBoxes([zeros(d) ones(d)], .1*ones(d), .1, coverage_fraction)
end

function RandomNonIntersectingBoxes(bounds, init, border, coverage_fraction)
    # RandomBoxInBounds(bnds) = bnds[:,1] .+ ((bnds[:,2] - bnds[:,1]) .* sort(rand(size(bnds, 1), 2), 2))
    function RandomBoxInBounds(bnds, min_vol_frac = .2)
        d = size(bnds, 1)
        box_unscaled = sort(rand(d, 2), 2)
        for i in 1:d
            if box_unscaled[i,2] - box_unscaled[i,1] < min_vol_frac^(1/d)
                if box_unscaled[i,1] + min_vol_frac^(1/d) > 1
                    box_unscaled[i,2] = .5 + .5*min_vol_frac^(1/d)
                    box_unscaled[i,1] = .5 - .5*min_vol_frac^(1/d)
                else
                    box_unscaled[i,2] = box_unscaled[i,1] + min_vol_frac^(1/d)
                end
            end
        end
        return bnds[:,1] .+ ((bnds[:,2] - bnds[:,1]) .* box_unscaled)
    end
    Volume(bnds) = prod(bnds[:,2] - bnds[:,1])

    d = length(init)
    boundsPQ = Collections.PriorityQueue()
    # for bnds in SplitRemainder([(bounds[:,1] + border) (bounds[:,2] - border)], [(init - border) (init + border)])
    for bnds in SplitRemainder(bounds, [(init - border/2) (init + border/2)])     # clamp general init/goal
        boundsPQ[bnds] = -Volume(bnds)
    end
    
    volume_goal = Volume(bounds)*coverage_fraction
    vol_so_far = 0.0;
    box_list = Matrix{Float64}[]
    while vol_so_far < volume_goal
        bnds = Collections.dequeue!(boundsPQ)
        while all(bnds[:,2] .> (bounds[:,2] - border))  # while loop should only proc once
            for b in SplitRemainder(bnds, [(bounds[:,2] - border) bounds[:,2]])    # should be general goal
                boundsPQ[b] = -Volume(b)
            end
            bnds = Collections.dequeue!(boundsPQ)
        end
        if Volume(bnds) < 0.001
            push!(box_list, bnds)
        else
            push!(box_list, RandomBoxInBounds(bnds))
            for b in SplitRemainder(bnds, box_list[end])
                boundsPQ[b] = -Volume(b)
            end
        end
        vol_so_far = vol_so_far + Volume(box_list[end])
        # println((vol_so_far, Volume(bnds)))
        # println(boundsPQ)
    end
    k = indmax(box_list[end][:,2] - box_list[end][:,1])
    box_list[end][k,2] = box_list[end][k,1] + (box_list[end][k,2] - box_list[end][k,1]) * (1 - (vol_so_far - volume_goal) / Volume(box_list[end]))

    return AABoxes(reshape(hcat(box_list...), d, 2, length(box_list)))
end

function SplitRemainder(bounds, bx)
    d = size(bounds, 1)
    if d == 1
        return Matrix{Float64}[[bounds[1] bx[1]], [bx[2] bounds[2]]]
    end
    # k = rand(1:d)
    k = indmax(bx[:,2] - bx[:,1])
    face1 = copy(bounds)
    face1[k,2] = bx[k,1]
    face2 = copy(bounds)
    face2[k,1] = bx[k,2]
    other_pieces =[copy(bounds) for i in 1:2*(d-1)]   # does fillcopy exist?
    for (i,v) in enumerate(SplitRemainder(bounds[1:d .!= k,:], bx[1:d .!= k,:]))
        other_pieces[i][1:d .!= k,:] = v
        other_pieces[i][k,:] = bx[k,:]
    end
    return [Matrix{Float64}[face1, face2], other_pieces]
end

function LoadObstacles(fname)
    obs_f = readlines(open(fname))
    dim = parseint(obs_f[1])
    raw_obs = [map(parsefloat, split(l)) for l in obs_f[2:end]]
    obs = Array(Float64, dim, 2, div(length(raw_obs), 2))
    for i in 1:int(length(raw_obs)/2)
        obs[:,1,i] = raw_obs[2i-1]
        obs[:,2,i] = raw_obs[2i]
    end
    return AABoxes(obs)
end

function LoadSamples(fname)
    V_f = readlines(open(fname))
    dim = parseint(V_f[1])
    raw_V = [map(parsefloat, split(l)) for l in V_f[2:end]]
    V = Array(Float64, dim, length(raw_V))
    for i in 1:length(raw_V)
        V[:,i] = raw_V[i]
    end
    return V
end