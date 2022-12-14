import cv2
import math
import numpy as np
import time
 
from pqdict import pqdict
from scipy.spatial import cKDTree
from State import State
from utils import draw_plan, compute_ndvol

class FMTPlanner:
    """
    Applies FMT* Algorithm on the given world
    """
    def __init__(
        self,
        world: np.ndarray,
        n_samples: int = 1000,
        max_iter: int = 10000,
        col_dst: np.float64 = 5.0,
        pr: np.float64 = 0.1,
        seed: int = 0) -> None:
        
        # RGB world is N-D with 3 channels for R, G and B
        self.rgbworld = world
        # World is n-dimensional array with 0: free, 1: occupied
        self.world = (self.rgbworld[:,:,0] == 0).astype(np.uint8)
        self.n = n_samples
        self.pr = pr
        self.max_iter = max_iter
        self.col_dst = col_dst
    
        self.d = len(self.world.shape)

        # Compute rn based on the definition in the paper
        self._compute_rn()

        # Construct the obstacle tree for collision check
        self.obstacle_tree = cKDTree(np.argwhere(self.world == 1))

        # Tracker for collision check calls
        self.cc_calls = 0
        # Currently, only uniform random sampling is implemented
        prng = np.random.RandomState(seed)
        self.node_list = list()
        while len(self.node_list) < self.n:
            # sample a d-dimensional State from a uniform distribution
            x_rand = int(prng.uniform(0, world.shape[0]))
            y_rand = int(prng.uniform(0, world.shape[1]))
            sample = State([x_rand, y_rand])
            # check if sample is in free space
            if self._is_collision_free(sample):
                self.node_list.append(sample)

    def _compute_rn(self, sf=1.5) -> None:
        lebesgue_xfree = np.bincount(self.world.ravel())[0]
        ndvol = compute_ndvol(1,self.d)
        gamma = sf*2*np.power((1/self.d)*(lebesgue_xfree/ndvol), 1/self.d)
        self.rn = np.round(gamma*np.power(np.log(self.n)/self.n, 1/self.d), decimals=2)

    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        curr_ptr = state
        path = [state]

        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        # return a reverse copy of the path (so that first state is starting state)
        return path[::-1]

    def _is_collision_free(self, s, t=None, searching=False) -> bool:
        if t is None or np.all(s == t):
            return self.obstacle_tree.query(s.v)[0] > self.col_dst
        if searching:
            self.cc_calls += 1

        ts_bar = t.v - s.v # d-dimensional 
        d = s.euclidean_distance(t)
        unit_ts_bar = ts_bar / d # d-dimensional unit-vector
        steps = np.arange(0, d+self.pr, self.pr).reshape(-1, 1)
        pts = s.v + steps * unit_ts_bar
        return bool(self.obstacle_tree.query(pts)[0].min() > self.col_dst)
    
    def compute_cost(self, plan):
        total_cost = 0.0
        for idx in range(1, len(plan)):
            total_cost += plan[idx].euclidean_distance(plan[idx-1])
        return total_cost

    def plan(self,
             start: np.ndarray,
             target: np.ndarray,
             map_idx: int,
             pidx: int,
             sidx: int,
             mode: str = 'test',
             hw: np.float64 = 0.0,
             showlive: bool = False) -> dict:
        """
        Run path planning

        Args:
            start (np.ndarray): Start location
            goal (np.ndarray): Goal location
            hw (int, optional): Weight for Euclidean heuristics. Defaults to 0.0.
        
        Returns:
            dict: Containing plan and number of steps required
        """
        path_found = False
        self.cc_calls = 0

        start = State(start)
        target = State(target)
        assert self._is_collision_free(start) and self._is_collision_free(target)

        self.node_list = [start] + self.node_list + [target]
        start_id = 0
        target_id = len(self.node_list)-1

        # init KDTree for nodes for an optimal search
        node_tree = cKDTree([s.v for s in self.node_list])
        # init euclidean heuristics for each node
        h = [s.euclidean_distance(target) for s in self.node_list]

        # Initialize V_open, V_unvisited and V_closed per the algorithm
        V_open = pqdict({start_id: 0.0}) # Add x_init
        V_unvisited = list(range(1,len(self.node_list)))
        V_closed = list()

        img = np.copy(self.rgbworld)
        if self.d == 2: 
            # Plot start and target
            cv2.circle(img, (start.v[1], start.v[0]), 6, (65, 200, 245), -1)
            cv2.circle(img, (target.v[1], target.v[0]), 6, (65, 245, 100), -1)
            # Plot the sampled points
            for i in range(1,len(self.node_list)-1):
                sample = self.node_list[i]
                cv2.circle(img, (sample.v[1], sample.v[0]), 2, (245, 95, 65))
            if (showlive):
                cv2.imshow('image', img)
        plan = [start]

        total_execution_time = 0.0
        # Begin
        for step in range(self.max_iter):
            iter_stime = time.time()
            z = V_open.top()
            # check if we have reached the goal
            if z == target_id:
                # print("Path found!!")
                path_found = True
                iter_etime = time.time()
                total_execution_time += (iter_etime - iter_stime)
                # store plan
                plan = self._follow_parent_pointers(self.node_list[z])
                break
            # If not, then expand the tree further
            # Query the Node KDTree for set of points with ball of rn
            N_z = node_tree.query_ball_point(self.node_list[z].v, self.rn)
            # Get set of points that are near z and in V_unvisited
            X_near = list(set(N_z) & set(V_unvisited))
            # For each of the above points find locally optimal connections to V_open
            for x in X_near:
                N_x = node_tree.query_ball_point(self.node_list[x].v, self.rn)
                Y_near = list(set(N_x) & set(V_open))
                # Find y in V_open with minimum cost-to-come from x
                # DP update: Cost(y,x) = rn as we use fixed ball radius for this implementation
                # This will change if were to use dynamic cost or K-NN variant of FMT*
                y_min = Y_near[np.argmin([V_open[y] for y in Y_near])]
                # Do collision check only for y_min ("lazy collision-check")
                if self._is_collision_free(self.node_list[x],
                                           self.node_list[y_min], searching=True):
                    # Add y_min as parent of x and x as child of y_min -- adding an edge in the tree
                    self.node_list[x].parent = self.node_list[y_min]
                    self.node_list[y_min].children.append(self.node_list[x])
                    # Add x (succesfully connected) to V_open with cost c_x
                    # print(h[x], ": ", type(h[x]))
                    c_x  = V_open[y_min] + self.node_list[y_min].euclidean_distance(self.node_list[x]) + hw*(-h[y_min] + h[x])
                    V_open.additem(x, c_x)
                    V_unvisited.remove(x)

                    iter_etime = time.time()
                    total_execution_time += (iter_etime - iter_stime)
                    # Update image with the newly added edge
                    cv2.line(img, (self.node_list[y_min].v[1], self.node_list[y_min].v[0]),
                                (self.node_list[x].v[1], self.node_list[x].v[0]),
                                (255,0,0)
                            )
                    iter_stime = time.time()
            # Move z from V_open to V_closed
            V_open.pop(z)
            V_closed.append(z)
            # Report failure if V_open is empty
            if len(V_open) == 0:
                # print("Path not found!")
                iter_etime = time.time()
                total_execution_time += (iter_etime - iter_stime)
                break

            iter_etime = time.time()
            total_execution_time += (iter_etime - iter_stime)
            
            if (showlive):
                # Show the updated image
                cv2.imshow('image', img)
                cv2.waitKey(1)
        
        draw_plan(img, plan, map_idx, sidx, self.n, pidx, hw, bgr=(0,0,255), thickness=2, mode=mode)

        return {
            'plan': plan,
            'path_found': path_found,
            'num_iters': step, # count
            'total_execution_time': np.round(total_execution_time*100, decimals=2), # in ms
            'collision_checks': self.cc_calls, # count
            'cost': np.round(self.compute_cost(plan), decimals=2) if path_found else -1
        }

