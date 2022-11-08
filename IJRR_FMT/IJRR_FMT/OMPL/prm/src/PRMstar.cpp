/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2013, Autonomous Systems Laboratory, Stanford University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of Stanford University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Authors: Ashley Clark (Stanford) and Wolfgang Pointner (AIT) */
/* Co-developers: Brice Rebsamen (Stanford) and Tim Wheeler (Stanford) */
/* Algorithm design: Lucas Janson (Stanford) and Marco Pavone (Stanford) */
/* Acknowledgements for insightful comments: Edward Schmerling (Stanford), Oren Salzman (Tel Aviv University), and Joseph Starek (Stanford) */

#include <limits>
#include <iostream>
#include <map>

#include <boost/math/constants/constants.hpp>
#include <boost/math/distributions/binomial.hpp>

#include <ompl/datastructures/BinaryHeap.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/geometric/planners/prm/PRMstar.h>


ompl::geometric::PRMstar::PRMstar(const base::SpaceInformationPtr &si)
    : base::Planner(si, "PRMstar")
    , numSamples_(1000)
    , radiusMultiplier_(1.1)
{
    // An upper bound on the free space volume is the total space volume; the free fraction is estimated in sampleFree
    freeSpaceVolume_ = si_->getStateSpace()->getMeasure();
    lastGoalMotion_ = NULL;
    collisionChecks_ = 0;
    nearestK_ = true;

    specs_.approximateSolutions = false;
    specs_.directed = false;

    ompl::base::Planner::declareParam<unsigned int>("num_samples", this, &PRMstar::setNumSamples, &PRMstar::getNumSamples, "10:10:100000");
    ompl::base::Planner::declareParam<double>("radius_multiplier", this, &PRMstar::setRadiusMultiplier, &PRMstar::getRadiusMultiplier, "0.1:0.05:50.");
    Planner::declareParam<bool>("nearest_k", this, &PRMstar::setNearestK, &PRMstar::getNearestK, "0,1");
}

ompl::geometric::PRMstar::~PRMstar()
{
    freeMemory();
}

void ompl::geometric::PRMstar::setup()
{
    Planner::setup();

    /* Setup the optimization objective. If no optimization objective was
       specified, then default to optimizing path length as computed by the
       distance() function in the state space */
    if (pdef_->hasOptimizationObjective())
        opt_ = pdef_->getOptimizationObjective();
    else
    {
        OMPL_INFORM("%s: No optimization objective specified. Defaulting to optimizing path length.", getName().c_str());
        opt_.reset(new base::PathLengthOptimizationObjective(si_));
    }
    H_.getComparisonOperator().opt_ = opt_.get();

    if (!nn_)
        nn_.reset(tools::SelfConfig::getDefaultNearestNeighbors<Motion*>(si_->getStateSpace()));
    nn_->setDistanceFunction(boost::bind(&PRMstar::distanceFunction, this, _1, _2));
}

void ompl::geometric::PRMstar::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion*> motions;
        motions.reserve(nn_->size());
        nn_->list(motions);
        for (unsigned int i = 0 ; i < motions.size() ; ++i)
        {
            si_->freeState(motions[i]->getState());
            delete motions[i];
        }
    }
}

void ompl::geometric::PRMstar::clear()
{
    Planner::clear();
    lastGoalMotion_ = NULL;
    sampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();
    H_.clear();
    hElements_.clear();
    neighborhoods_.clear();

    collisionChecks_ = 0;
}

void ompl::geometric::PRMstar::getPlannerData(base::PlannerData &data) const
{
    Planner::getPlannerData(data);
    std::vector<Motion*> motions;
    nn_->list(motions);

    if (lastGoalMotion_)
        data.addGoalVertex(base::PlannerDataVertex(lastGoalMotion_->getState()));

    unsigned int size = motions.size();
    for (unsigned int i = 0; i < size; ++i)
    {
        if (motions[i]->getParent() == NULL)
            data.addStartVertex(base::PlannerDataVertex(motions[i]->getState()));
        else
            data.addEdge(base::PlannerDataVertex(motions[i]->getParent()->getState()),
                         base::PlannerDataVertex(motions[i]->getState()));
    }
}

void ompl::geometric::PRMstar::saveNeighborhood(Motion *m)
{
    // Check to see if neighborhood has not been saved yet
    if (neighborhoods_.find(m) == neighborhoods_.end())
    {
        std::vector<Motion*> nbh;
        if (nearestK_)
            nn_->nearestK(m, NNk, nbh);
        else
            nn_->nearestR(m, NNr, nbh);
        if (!nbh.empty())
        {
            // Save the neighborhood but skip the first element, since it will be motion m
            neighborhoods_[m] = std::vector<Motion*>(nbh.size() - 1, 0);
            std::copy(nbh.begin() + 1, nbh.end(), neighborhoods_[m].begin());
        }
        else
        {
            // Save an empty neighborhood
            neighborhoods_[m] = std::vector<Motion*>(0);
        }
    } // If neighborhood hadn't been saved yet
}

// Calculate the unit ball volume for a given dimension
double ompl::geometric::PRMstar::calculateUnitBallVolume(const unsigned int dimension) const
{
    if (dimension == 0)
        return 1.0;
    else if (dimension == 1)
        return 2.0;
    return 2.0 * boost::math::constants::pi<double>() / dimension
            * calculateUnitBallVolume(dimension - 2);
}

double ompl::geometric::PRMstar::calculateRadius(const unsigned int dimension, const unsigned int n) const
{
    double a = 1.0 / (double)dimension;
    double unitBallVolume = calculateUnitBallVolume(dimension);

    return radiusMultiplier_ * 2.0 * std::pow(a, a) * std::pow(freeSpaceVolume_ / unitBallVolume, a) * std::pow(log((double)n) / (double)n, a);
}

void ompl::geometric::PRMstar::sampleFree(const base::PlannerTerminationCondition &ptc)
{
    unsigned int nodeCount = 0;
    unsigned int sampleAttempts = 0;
    Motion *motion = new Motion(si_);

    // Sample numSamples_ number of nodes from the free configuration space
    while (nodeCount < numSamples_ && !ptc)
    {
        sampler_->sampleUniform(motion->getState());
        sampleAttempts++;

        bool collision_free = si_->isValid(motion->getState());

        if (collision_free)
        {
            nodeCount++;
            nn_->add(motion);
            motion = new Motion(si_);
        } // If collision free
    } // While nodeCount < numSamples
    si_->freeState(motion->getState());
    delete motion;

    // 95% confidence limit for an upper bound for the true free space volume
    freeSpaceVolume_ = boost::math::binomial_distribution<>::find_upper_bound_on_p(sampleAttempts, nodeCount, 0.05) * si_->getStateSpace()->getMeasure();
}

void ompl::geometric::PRMstar::assureGoalIsSampled(const ompl::base::GoalSampleableRegion *goal)
{
    // Ensure that there is at least one node near each goal
    while (const base::State *goalState = pis_.nextGoal())
    {
        Motion *gMotion = new Motion(si_);
        si_->copyState(gMotion->getState(), goalState);

        std::vector<Motion*> nearGoal;
        nn_->nearestR(gMotion, goal->getThreshold(), nearGoal);

        // If there is no node in the goal region, insert one
        if (nearGoal.empty())
        {
            OMPL_DEBUG("No state inside goal region");
            if (si_->getStateValidityChecker()->isValid(gMotion->getState()))
            {
                gMotion->setSetType(Motion::SET_W);
                nn_->add(gMotion);
            }
            else
            {
                si_->freeState(gMotion->getState());
                delete gMotion;
            }
        }
        else // There is already a sample in the goal region
        {
            si_->freeState(gMotion->getState());
            delete gMotion;
        }
    } // For each goal
}

void ompl::geometric::PRMstar::propagateCosts(Motion* node) {
    if (node) {
        // use binary heap in combination with map to propagate costs in graph
        ompl::BinaryHeap<Motion*, MotionCompare> binHeap;
        std::map<Motion*, ompl::BinaryHeap<Motion*, MotionCompare>::Element*> heapElements;
        binHeap.getComparisonOperator().opt_ = opt_.get();

        // add initial node to tree and store it and the heap element in map
        heapElements[node] =  binHeap.insert(node);

        // take head from min heap until it's empty
        while(binHeap.size() > 0)
        {
            //remove head from heap
            Motion* u = binHeap.top()->data;
            binHeap.pop();

            // add all its children with costs to the heap
            for(unsigned int i = 0; i < u->children.size(); i++)
            {
                Motion *v = u->children.at(i);

                base::Cost cnew = opt_->motionCost(u->getState(), v->getState());
                base::Cost alternative = opt_->combineCosts(u->getCost(), cnew);

                if(opt_->isCostBetterThan(alternative, v->getCost()))
                {
                    v->setCost(alternative);
                    v->setParent(u);


                    //check if node is already an element of the heap
                    if(heapElements[v] != NULL)
                    {
                        // update the existing heap element
                        binHeap.update(heapElements[v]);
                    }
                    else
                    {
                        // add new element to heap and also store it in map
                        // heapElements[v] = binHeap.insert(v);
                        ompl::BinaryHeap<Motion*, MotionCompare>::Element* vElement = binHeap.insert(v);
                        heapElements[v] = vElement;
                    }
                }
            }
        }

        binHeap.clear();
        heapElements.clear();
    }
}


ompl::base::PlannerStatus ompl::geometric::PRMstar::solve(const base::PlannerTerminationCondition &ptc)
{
    if (lastGoalMotion_) {
        OMPL_INFORM("solve() called before clear(); returning previous solution");
        traceSolutionPathThroughTree(lastGoalMotion_);
        OMPL_DEBUG("Final path cost: %f", lastGoalMotion_->getCost().v);
        return base::PlannerStatus(true, false);
    }
    else if (hElements_.size() > 0)
    {
        OMPL_INFORM("solve() called before clear(); no previous solution so starting afresh");
        clear();
    }

    checkValidity();
    base::GoalSampleableRegion *goal = dynamic_cast<base::GoalSampleableRegion*>(pdef_->getGoal().get());
    Motion *initMotion = NULL;

    if (!goal)
    {
        OMPL_ERROR("%s: Unknown type of goal", getName().c_str());
        return base::PlannerStatus::UNRECOGNIZED_GOAL_TYPE;
    }

    // Add start states to V (nn_) and H
    while (const base::State *st = pis_.nextStart())
    {
        initMotion = new Motion(si_);
        si_->copyState(initMotion->getState(), st);
        hElements_[initMotion] = H_.insert(initMotion);
        initMotion->setSetType(Motion::SET_H);
        initMotion->setCost(opt_->initialCost(initMotion->getState()));
        nn_->add(initMotion); // V <-- {x_init}
    }

    if (!initMotion)
    {
        OMPL_ERROR("Start state undefined");
        return base::PlannerStatus::INVALID_START;
    }

    // Sample N free states in the configuration space
    if (!sampler_)
        sampler_ = si_->allocStateSampler();
    sampleFree(ptc);
    assureGoalIsSampled(goal);
    OMPL_INFORM("%s: Starting planning with %u states already in datastructure", getName().c_str(), nn_->size());

    // Calculate the nearest neighbor search radius
    if (nearestK_) {
        NNk = std::ceil(std::pow(2.0 * radiusMultiplier_, (double)si_->getStateDimension()) *
                        (boost::math::constants::e<double>() / (double)si_->getStateDimension()) *
                        log((double)nn_->size()));
        OMPL_DEBUG("Using nearest-neighbors k of %d", NNk);
    } else {
        NNr = calculateRadius(si_->getStateDimension(), nn_->size());
        OMPL_DEBUG("Using radius of %f", NNr);
    }

    // Flag all nodes as in set W
    std::vector<Motion*> vNodes;
    vNodes.reserve(nn_->size());
    nn_->list(vNodes);
    unsigned int vNodesSize = vNodes.size();
    for (unsigned int i = 1; i < vNodesSize; ++i)
    {
        vNodes[i]->setCost(base::Cost(std::numeric_limits<double>::infinity()));
        vNodes[i]->idx = i;     // to avoid double checking u->v and vice versa
    }
    vNodes[0]->idx = 0;

    // PRM diverges from FMT
    unsigned int i = 0;
    Motion* v;
    // iterate through all elements of V
    while(i < vNodes.size() && !ptc)
    {
        v = vNodes.at(i);

        std::vector<Motion*> U;
        if (nearestK_)
            nn_->nearestK(v, NNk, U);
        else
            nn_->nearestR(v, NNr, U);

        if(U.size() > 0)
        {
            U.erase(U.begin(), U.begin()+1);        // exclude v from U , is always first element in neighborhood
        }

        for(unsigned int j = 0; j < U.size(); j++)
        {
            Motion* u = U.at(j);
            if (v->idx < u->idx) {
                // check if motion from v to u is collision free

                ++collisionChecks_;
                bool collision_free = si_->checkMotion(v->getState() , u->getState()); // collision checking

                if (collision_free) {
                    // add edge between nodes
                    u->children.push_back(v);
                    v->children.push_back(u);
                    // std::vector<double> ureals = ob::ScopedState<>(si_->getStateSpace(), u->getState()).reals();
                    // std::vector<double> vreals = ob::ScopedState<>(si_->getStateSpace(), v->getState()).reals();
                    // OMPL_DEBUG("Adding edge between: %f %f, and %f %f\n",ureals[0],ureals[1],vreals[0],vreals[1]);
                }
            }
        }
        i++;
    }
    
    propagateCosts(vNodes.at(0));

    lastGoalMotion_ = NULL;
    base::Cost best_cost = base::Cost(std::numeric_limits<double>::infinity());
    for (unsigned int i = 0; i < vNodesSize; ++i)
    {
        if (goal->isSatisfied(vNodes[i]->getState()) && opt_->isCostBetterThan(vNodes[i]->getCost(), best_cost))
        {
            best_cost = vNodes[i]->getCost();
            lastGoalMotion_ = vNodes[i];
        }
    }

    if(!ptc && lastGoalMotion_ != NULL)
    {
        traceSolutionPathThroughTree(lastGoalMotion_);
       
        OMPL_DEBUG("Final path cost: %f\n", lastGoalMotion_->getCost().v);
        // A solution has been found iff the path starts at the start state
        return base::PlannerStatus(true, false);
    }
    else
    {
        OMPL_DEBUG("No solution could be found.");
        //planner terminated without accomplishing goal
        return base::PlannerStatus(false, false);
    }
}

void ompl::geometric::PRMstar::traceSolutionPathThroughTree(Motion *goalMotion)
{
    std::vector<Motion*> mpath;
    Motion *solution = goalMotion;

    // Construct the solution path
    while (solution != NULL)
    {
        mpath.push_back(solution);
        solution = solution->getParent();
    }

    // Set the solution path
    PathGeometric *path = new PathGeometric(si_);
    int mPathSize = mpath.size();
    for (int i = mPathSize - 1 ; i >= 0 ; --i)
        path->append(mpath[i]->getState());
    pdef_->addSolutionPath(base::PathPtr(path), false, lastGoalMotion_->getCost().v, getName());
}

bool ompl::geometric::PRMstar::expandTreeFromNode(Motion *&z)
{
    // Find all nodes that are near z, and also in set W

    std::vector<Motion*> xNear;
    const std::vector<Motion*> &zNeighborhood = neighborhoods_[z];
    unsigned int zNeighborhoodSize = zNeighborhood.size();
    xNear.reserve(zNeighborhoodSize);

    for (unsigned int i = 0; i < zNeighborhoodSize; ++i)
    {
        Motion *x = zNeighborhood[i];
        if (x->getSetType() == Motion::SET_W) {
            saveNeighborhood(x);
            if (nearestK_) {
                // Relies on NN datastructure returning k-nearest in sorted order
                if (opt_->motionCost(z->getState(), x->getState()).v <= opt_->motionCost(neighborhoods_[x].back()->getState(), x->getState()).v)
                    xNear.push_back(zNeighborhood[i]);
            } else {
                xNear.push_back(x);
            }
        }
    }

    // For each node near z and in set W, attempt to connect it to set H
    std::vector<Motion*> yNear;
    std::vector<Motion*> H_new;
    unsigned int xNearSize = xNear.size();
    for (unsigned int i = 0 ; i < xNearSize; ++i)
    {
        Motion *x = xNear[i];

        // Find all nodes that are near x and in set H
        saveNeighborhood(x);
        const std::vector<Motion*> &xNeighborhood = neighborhoods_[x];

        unsigned int xNeighborhoodSize = xNeighborhood.size();
        yNear.reserve(xNeighborhoodSize);
        for (unsigned int j = 0; j < xNeighborhoodSize; ++j)
        {
            if (xNeighborhood[j]->getSetType() == Motion::SET_H)
                yNear.push_back(xNeighborhood[j]);
        }

        // Find the lowest cost-to-come connection from H to x
        Motion *yMin = NULL;
        base::Cost cMin(std::numeric_limits<double>::infinity());
        unsigned int yNearSize = yNear.size();
        for (unsigned int j = 0; j < yNearSize; ++j)
        {
            base::State *yState = yNear[j]->getState();
            base::Cost dist = opt_->motionCost(yState, x->getState());
            base::Cost cNew = opt_->combineCosts(yNear[j]->getCost(), dist);

            if (opt_->isCostBetterThan(cNew, cMin))
            {
                yMin = yNear[j];
                cMin = cNew;
            }
        }
        yNear.clear();

        // If an optimal connection from H to x was found
        if (yMin != NULL)
        {
            ++collisionChecks_;
            bool collision_free = si_->checkMotion(yMin->getState(), x->getState());

            if (collision_free)
            {
                // Add edge from yMin to x
                x->setParent(yMin);
                x->setCost(cMin);
                // Add x to H_new
                H_new.push_back(x);
                // Remove x from W
                x->setSetType(Motion::SET_NULL);
            }
        } // An optimal connection from H to x was found
    } // For each node near z and in set W, try to connect it to set H

    // Remove motion z from the binary heap and from the map
    H_.remove(hElements_[z]);
    hElements_.erase(z);
    z->setSetType(Motion::SET_NULL);

    // Add the nodes in H_new to H
    unsigned int hNewSize = H_new.size();
    for (unsigned int i = 0; i < hNewSize; i++)
    {
        hElements_[H_new[i]] = H_.insert(H_new[i]);
        H_new[i]->setSetType(Motion::SET_H);
    }

    H_new.clear();

    if (H_.empty())
    {
        OMPL_INFORM("H is empty before path was found --> no feasible path exists");
        return false;
    }

    // Take the top of H as the new z
    z = H_.top()->data;

    return true;
}

std::string ompl::geometric::PRMstar::getCollisionCheckCount() const
{
  return boost::lexical_cast<std::string>(collisionChecks_);
}