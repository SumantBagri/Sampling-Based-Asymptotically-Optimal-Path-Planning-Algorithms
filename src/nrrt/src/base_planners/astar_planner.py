#!/usr/bin/python
import sys
import time
import pickle
import numpy as np
from itertools import product
from math import cos, sin, pi, sqrt

from priority_queue import priority_dict

class State:
    """
    2D state.
    """

    def __init__(self, x, y):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y


    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))

    def __lt__(self, other):
        """
        Comparison operator to break ties when two states have the same priority.
        """
        return self.x < other.x or self.y < other.y

    def __repr__(self):
        return "({}, {})".format(self.x, self.y)

class AStarPlanner:
    """
    Applies the A* shortest path algorithm on a given grid world
    """

    def __init__(self, world, clearance=1):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world
        self.occ_grid = (self.occ_grid == 0).astype('uint8')
        self.clearance = clearance

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-self.clearance:state.y+self.clearance, state.x-self.clearance:state.x+self.clearance] == 0).all()

    def get_neighboring_states(self, state):
        """
        Returns free neighboring states of the given state. Returns up to 8
        neighbors (north, south, east, west, northwest, northeast, southwest, southeast)
        """

        x = state.x
        y = state.y

        rows, cols = self.world.shape[:2]

        dx = [0]
        dy = [0]

        if (x > 0):
            dx.append(-1)

        if (x < rows -1):
            dx.append(1)

        if (y > 0):
            dy.append(-1)

        if (y < cols -1):
            dy.append(1)

        # product() returns the cartesian product
        # yield is a python generator. Look it up.
        for delta_x, delta_y in product(dx,dy):
            if delta_x != 0 or delta_y != 0:
                ns = State(x + delta_x, y + delta_y)
                if self.state_is_free(ns):
                    yield ns


    def _follow_parent_pointers(self, parents, state):
        """
        Assumes parents is a dictionary. parents[key]=value
        means that value is the parent of key. If value is None
        then key is the starting state. Returns the shortest
        path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        assert (state in parents)
        curr_ptr = state
        shortest_path = []

        while curr_ptr is not None:
            shortest_path.append((curr_ptr.x, curr_ptr.y))
            curr_ptr = parents[curr_ptr]

        # return a reverse copy of the path (so that first state is starting state)
        return shortest_path[::-1]

    def compute_distance(self, u, v, dist_type="chebyshev"):
        if dist_type == "chebyshev":
            return max(u.x - v.x, u.y - v.y)
        elif dist_type == "euclidean":
            return sqrt((u.x - v.x)**2 + (u.y - v.y)**2)

    # TODO: this method currently has the implementation of Dijkstra's algorithm.
    # Modify it to implement A*. The changes should be minor.
    def plan(self, start_state, dest_state, algo="astar"):
        """
        Returns the shortest path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # Q is a mutable priority queue implemented as a dictionary
        Q = priority_dict()
        Q[start_state] = 0.0

        # Array that contains the optimal distance to come from the starting state
        dist_to_come = float("inf") * np.ones((self.world.shape[0], self.world.shape[1]))
        dist_to_come[start_state.x, start_state.y] = 0

        # Boolean array that is true iff the distance to come of a state has been
        # finalized
        evaluated = np.zeros((self.world.shape[0], self.world.shape[1]), dtype='uint8')

        # Contains key-value pairs of states where key is the parent of the value
        # in the computation of the shortest path
        parents = {start_state: None}
        
        # keep track of visited states, total distance to dest
        self.visited = []
        self.shortest_dist = 0

        while Q:
            # s is also removed from the priority Q with this
            s = Q.pop_smallest()

            # Assert s hasn't been evaluated before
            assert (evaluated[s.x, s.y] == 0)
            evaluated[s.x, s.y] = 1
            self.visited.append((s.x, s.y))

            if s == dest_state:
                return self._follow_parent_pointers(parents, s)
                
            # for all free neighboring states
            for ns in self.get_neighboring_states(s):
                if evaluated[ns.x, ns.y] == 1:
                    continue
                
                # cost-to-come
                transition_distance = sqrt((ns.x - s.x)**2 + (ns.y - s.y)**2)
                alternative_dist_to_come_to_ns = dist_to_come[s.x, s.y] + transition_distance

                # adding heuristic component for astar
                if algo == "astar":
                    heuristic = self.compute_distance(s, ns, dist_type="euclidean")
                    alternative_dist_to_come_to_ns = alternative_dist_to_come_to_ns + heuristic

                # if the state ns has not been visited before or we just found a shorter path
                # to visit it then update its priority in the queue, and also its
                # distance to come and its parent
                if (ns not in Q) or (alternative_dist_to_come_to_ns < dist_to_come[ns.x, ns.y]):
                    Q[ns] = alternative_dist_to_come_to_ns
                    dist_to_come[ns.x, ns.y] = alternative_dist_to_come_to_ns
                    parents[ns] = s

        return [start_state.x, start_state.y]