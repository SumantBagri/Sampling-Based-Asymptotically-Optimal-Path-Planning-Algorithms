#!/usr/bin/python
import time
import numpy as np
import random
import cv2
import heapq

from math import cos, sin, pi, sqrt, acos, log

from plotting_utils import draw_plan
from queue import Queue

from state import State


class BITPlanner:
    """
    Applies the BIT algorithm on a given grid world
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:, :, 0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

        self.current_path_cost_arr = []
        self.current_time_elapsed_arr = []
        self.current_time_ex_plotting_elapsed_arr = []
        self.total_plotting_time = 0
        self.time_tracker = dict()

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-1:state.y+1, state.x-1:state.x+1] == 0).all()

    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image.
        """
        # TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
        x = random.randrange(self.world.shape[1])
        y = random.randrange(self.world.shape[0])
        state = State(x, y, None)

        # Make sure the sampled state is not occupied
        while not (self.state_is_free(state)):
            x = random.randrange(self.world.shape[1])
            y = random.randrange(self.world.shape[0])
            state = State(x, y, None)

        return state

    def sample_batch(self, m, current_cost_to_goal, start_state, dest_state, img):
        """
        Sample a batch of new states uniformly randomly on the image.
        m : batch size to sample
        current_cost_to_goal : cost to come to the goal state from the start state given the current tree
        """
        start_time = time.time()
        sample_batch_time = 0
        plot_time = 0
        batch = []

        while len(batch) < m:
            s_rand = self.sample_state()
            # Estimated cost of path from start to goal that passes through this state
            est_cost_to_goal = self.est_cost_to_come(
                start_state, s_rand) + self.est_cost_to_go(dest_state, s_rand)
            if est_cost_to_goal < current_cost_to_goal:
                batch.append(s_rand)
                plot_start_time = time.time()
                cv2.circle(img, (s_rand.x, s_rand.y), 2, (0, 0, 0))
                plot_time = plot_time + (time.time() - plot_start_time)

        sample_batch_time = time.time() - start_time - plot_time
        self.time_tracker["SampleBatch"] = sample_batch_time
        self.total_plotting_time = self.total_plotting_time + plot_time

        return batch

    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        curr_ptr = state
        path = [state]  # Change to [] ?

        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        # return a reverse copy of the path (so that first state is starting state)
        return path[::-1]

    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state

    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:

        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y

        Otherwise, s_rand is farther than max_radius from s_nearest.
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.

        """

        # TODO: populate x and y properly according to the description above.
        # Note: x and y are integers and they should be in {0, ..., cols -1}
        # and {0, ..., rows -1} respectively
        euc_dist = s_nearest.euclidean_distance(s_rand)
        if euc_dist <= max_radius:
            x = s_rand.x
            y = s_rand.y
        else:
            cos_theta = (s_rand.x - s_nearest.x)/euc_dist
            sin_theta = (s_rand.y - s_nearest.y)/euc_dist

            delta_x = cos_theta * max_radius
            delta_y = sin_theta * max_radius

            x = int(s_nearest.x + delta_x)
            y = int(s_nearest.y + delta_y)

        # Make sure that the new coordinates are within the world
        # Not sure if this is necessary because s_rand should already be checked, but just in case
        if x > self.world.shape[1]:
            x = self.world.shape[1]
        if y > self.world.shape[0]:
            y = self.world.shape[0]
        if x < 0:
            x = 0
        if y < 0:
            y = 0

        s_new = State(x, y, s_nearest)
        return s_new

    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False

        max_checks = 50
        for i in range(max_checks):
            # TODO: check if the inteprolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False
            h = float(i)/max_checks
            x = int(s_from.x + h * (s_to.x - s_from.x))  # x + delta_x
            y = int(s_from.y + h * (s_to.y - s_from.y))  # y + delta_y
            s_interpolated = State(x, y, None)
            if (not (self.state_is_free(s_interpolated))):
                return False

        # Otherwise the line is free, so return true
        return True

    # Algorithm 3
    def prune(self, start_state, dest_state, max_cost, V, E, X_samples):
        # Do not need to prune X samples, they are already pruned

        start_time = time.time()

        # Prune vertices
        for v in V.copy():
            est_cost_to_goal = self.est_cost_to_come(
                start_state, v) + self.est_cost_to_go(dest_state, v)
            if est_cost_to_goal > max_cost:
                V.remove(v)

        # Prune edges <--- Look into this, why not being done for states Data Structure? <--- Dont think it's needed, since it doesn't explicitly define edges
        for e in E.copy():
            v1 = e[0]
            v2 = e[1]

            est_cost_to_goal_v1 = self.est_cost_to_come(
                start_state, v1) + self.est_cost_to_go(dest_state, v1)
            est_cost_to_goal_v2 = self.est_cost_to_come(
                start_state, v2) + self.est_cost_to_go(dest_state, v2)

            # <--- Look into this, Not being done often <--- Looks okay, edges also being removed somewhere else
            if est_cost_to_goal_v1 > max_cost or est_cost_to_goal_v2 > max_cost:
                E.remove(e)

        # Prune vertices in the states data structure
        states_to_open = Queue(maxsize=0)
        states_to_open.put(start_state)
        while not (states_to_open.empty()):  # While Q not empty
            # Pop next state to open from Q
            state = states_to_open.get()

            # For each child of this state, check its est_cost_to_goal
            # If it is greater than the max_cost, remove it from the children list
            # Else add it to the Q
            for child in state.children:
                est_cost_to_goal = self.est_cost_to_come(
                    start_state, child) + self.est_cost_to_go(dest_state, child)
                if est_cost_to_goal > max_cost:
                    state.children.remove(child)
                else:
                    states_to_open.put(child)

        # Move vertices with gT(v) = inf to X_samples (Vertices that were previously
        # connected to the tree but no longer are)
        for v in V.copy():
            if v.current_cost_to_come == float("Inf"):
                X_samples.add(v)
                V.remove(v)

        self.time_tracker["Prune"] = self.time_tracker["Prune"] + \
            (time.time() - start_time)

    # Admissible estimate of cost to come to current state from start state
    def est_cost_to_come(self, start_state, state):
        return state.euclidean_distance(start_state)

    # Admissible estimate of cost to go from current state to dest_state
    def est_cost_to_go(self, dest_state, state):
        return state.euclidean_distance(dest_state)

    # Admissible estimate of cost of edge (does not account for obstacles)
    def est_edge_cost(self, state1, state2):
        return state1.euclidean_distance(state2)

    def edge_cost(self, state1, state2):
        start_time = time.time()
        if self.path_is_obstacle_free(state1, state2):
            self.time_tracker["EdgeCostCalc"] = self.time_tracker["EdgeCostCalc"] + \
                (time.time() - start_time)
            return state1.euclidean_distance(state2)
        else:
            self.time_tracker["EdgeCostCalc"] = self.time_tracker["EdgeCostCalc"] + \
                (time.time() - start_time)
            return float("Inf")

    # Radius r of underlying RGG
    def radius(self, start_state, dest_state, eta, q, cost_best):

        n = 2  # Assuming it is dimensions

        # In 2D, the Lebesgue measure of a set is its area
        # The set Xf is an ellipse, the area of an ellipse is pi*a*b
        a = start_state.euclidean_distance(dest_state)

        x = a/2
        h = cost_best/2
        y = sin(acos(x/h)) * h

        b = y*2

        lebesgue_Xf = pi*a*b

        # The Lebesgue measure of a 2D unit ball is pi*r^2 = pi
        lebesgue_unit_ball = pi

        term1 = 2*eta
        term2 = (1+(1/n))**(1/n)
        term3 = (lebesgue_Xf/lebesgue_unit_ball)**(1/n)
        term4 = (log(q)/q)**(1/n)

        return term1*term2*term3*term4

    def add_states_to_Q_V(self, dest_state, Q_V, states):

        start_time = time.time()

        for state in states:
            current_est_cost = state.current_cost_to_come + \
                self.est_cost_to_go(dest_state, state)
            heapq.heappush(Q_V, (current_est_cost, state))

        self.time_tracker["AddStatesToV"] = self.time_tracker["AddStatesToV"] + \
            (time.time() - start_time)

    # Algorithm 2
    def expand_top_vertex(self, Q_V, Q_E, X_samples, V, E, r, V_old, start_state, dest_state):
        start_time = time.time()
        priority_cost, state = heapq.heappop(Q_V)
        #print("r = ", r)

        # Explore the edges to the nearby sampled states (by adding them to the edge queue Q_E)
        X_near = [
            X_sample for X_sample in X_samples if X_sample.euclidean_distance(state) <= r]

        for x_near in X_near:  # O(X_samples)
            # Only estimates (can't be infinity)
            est_cost = self.est_cost_to_come(start_state, state) + \
                self.est_edge_cost(state, x_near) + \
                self.est_cost_to_go(dest_state, x_near)
            if est_cost < dest_state.current_cost_to_come:
                # This one uses the current cost of coming to vertex state
                current_est_cost = state.current_cost_to_come + \
                    self.est_edge_cost(state, x_near) + \
                    self.est_cost_to_go(dest_state, x_near)
                heapq.heappush(Q_E, (current_est_cost, (state, x_near)))

        if state not in V_old:
            # Explore the edges to the nearby vertices (by adding them to the edge queue Q_E)
            V_near = [
                v for v in V if v.euclidean_distance(state) <= r]

            for v_near in V_near:  # O(V)
                # Only estimates (can't be infinity)
                est_cost = self.est_cost_to_come(
                    state) + self.est_edge_cost(state, v_near) + self.est_cost_to_go(v_near)

                cond1 = est_cost < dest_state.current_cost_to_come
                cond2 = state.current_cost_to_come + \
                    self.est_edge_cost(
                        state, v_near) < v_near.current_cost_to_come
                cond3 = (state, v_near) not in E

                if cond1 and cond2 and cond3:
                    # This one uses the current cost of coming to vertex state
                    current_est_cost = state.current_cost_to_come + \
                        self.est_edge_cost(state, v_near) + \
                        self.est_cost_to_go(v_near)
                    heapq.heappush(Q_E, (current_est_cost, (state, v_near)))

        self.time_tracker["ExpandV"] = self.time_tracker["ExpandV"] + \
            (time.time() - start_time)
        return

    # Algorithm 1
    def plan(self, start_state, dest_state, max_num_steps, batch_size, run_num, imgs_path, eta=2, show_img=True):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        start_time = time.time()
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # Initialize everything
        start_state.current_cost_to_come = 0
        V = set()
        V.add(start_state)
        E = set()
        X_samples = set()
        X_samples.add(dest_state)
        Q_E = []  # Priority Queue Implemented as Heap
        Q_V = []  # Priority Queue Implemented as Heap
        r = float("Inf")
        # Image to be used to display the tree
        img = np.copy(self.world)
        plan = [start_state]  # Final play for display
        current_batch_size = batch_size[0]

        # Reset Evaluation Metric Collection
        self.current_iteration_arr = []
        self.current_path_cost_arr = []
        self.any_path_found_arr = []
        self.prev_batch_size = 0
        self.batch_size_arr = []
        self.num_collision_checks_arr = []
        self.num_collision_checks = 0
        self.cumulative_sampled_arr = []
        self.cumulative_sampled = 0

        # Evaluation Metrics - Time Related
        # Maybe add Heap pushing time too?
        self.current_time_elapsed_arr = []
        self.current_time_ex_plotting_elapsed_arr = []
        self.total_plotting_time = 0
        self.time_tracker["Prune"] = 0
        self.time_tracker["AddStatesToV"] = 0
        self.time_tracker["ExpandV"] = 0
        self.time_tracker["EdgeCostCalc"] = 0
        self.time_tracker["RemoveFromE"] = 0
        self.time_tracker["RemoveFromQE"] = 0
        self.time_tracker["SampleBatch"] = 0
        self.time_tracker["TotalExPlotting"] = time.time(
        ) - start_time - self.total_plotting_time

        # Repeat until STOP
        for step in range(max_num_steps):

            print("Step: ", step)

            if len(Q_E) == 0 and len(Q_V) == 0:
                self.prune(start_state, dest_state,
                           dest_state.current_cost_to_come, V, E, X_samples)  # O(E)
                X_samples.update(self.sample_batch(
                    current_batch_size, dest_state.current_cost_to_come, start_state, dest_state, img))  # O(current_batch_size) I think
                self.cumulative_sampled += current_batch_size
                V_old = V
                self.add_states_to_Q_V(dest_state, Q_V, V)  # O(V)
                r = self.radius(start_state, dest_state, eta, len(V)+len(X_samples),
                                dest_state.current_cost_to_come)  # O(1)
                # Increment batch size unless it is already >= the max
                self.prev_batch_size = current_batch_size
                if (current_batch_size + batch_size[1] <= batch_size[2]):
                    current_batch_size = current_batch_size + batch_size[1]

            # Q_E/V[0][0] is equivalent to Q_E is equivalent to BestQueueValue(Q_E/V)
            # because it is implemented as a Heap (top item has min value, 0th item is the value)
            # Added (len(Q_E) == 0) because it can't compare until Q_E has something in it
            # Added (len(Q_V) != 0) because otherwise there isn't anything in the Q_V queue to compare with
            while (len(Q_V) != 0) and (len(Q_E) == 0 or Q_V[0][0] <= Q_E[0][0]):
                self.expand_top_vertex(
                    Q_V, Q_E, X_samples, V, E, r, V_old, start_state, dest_state)  # O(max(V, X_samples))

            # BestInQueue(Q_E)
            best_Q_E_cost, best_Q_E = heapq.heappop(Q_E)
            v_m, x_m = best_Q_E

            cond1 = (v_m.current_cost_to_come + self.est_edge_cost(v_m, x_m) +
                     self.est_cost_to_go(dest_state, x_m)) < dest_state.current_cost_to_come
            if cond1:
                # Actually do edge cost calculation (expensive operation)
                # Can be used multiple times
                edge_cost = self.edge_cost(v_m, x_m)  # O(max_checks=200)
                self.num_collision_checks += 1
                cond2 = (self.est_cost_to_come(start_state, v_m) + edge_cost +
                         self.est_cost_to_go(dest_state, x_m)) < dest_state.current_cost_to_come
                if cond2:
                    # In the first iteration, the current_cost_to_come of x_m is infinity, so
                    # this condition is True by default
                    x_m_new_cost_to_come = v_m.current_cost_to_come + edge_cost
                    cond3 = x_m_new_cost_to_come < x_m.current_cost_to_come
                    if cond3:
                        # This is where the cost is updated
                        x_m.current_cost_to_come = x_m_new_cost_to_come

                        if x_m in V:
                            # Remove the edge that X_m is previously connected to
                            current_time = time.time()
                            for (v1, v2) in E:  # O(E)
                                if v2 == x_m:
                                    E.remove((v1, v2))
                                    self.time_tracker["RemoveFromE"] = self.time_tracker["RemoveFromE"] + (
                                        time.time() - current_time)
                                    break  # There should only be one
                        else:
                            # Added this condition myself, otherwise it does not improve the original path
                            if x_m != dest_state:
                                # New vertex, remove from samples, add to V and vertex expansion queue Q_V
                                X_samples.remove(x_m)
                            # Moved to outside the condition (Didn't seem to make a difference in performance)
                            V.add(x_m)
                            heapq.heappush(
                                Q_V, (x_m.current_cost_to_come, x_m))

                        # Add the new edge to the graph
                        E.add((v_m, x_m))
                        v_m.children.append(x_m)
                        x_m.parent = v_m
                        # Remove the edges no longer better than current edge from edge expansion queue Q_E
                        current_time = time.time()
                        for q_e in Q_E:  # O(Q_E)
                            v1, v2 = q_e[1]
                            if v2 == x_m:
                                cost_through_v1 = v1.current_cost_to_come + \
                                    self.est_edge_cost(v1, x_m)
                                if cost_through_v1 >= x_m.current_cost_to_come:
                                    Q_E.remove(q_e)
                        self.time_tracker["RemoveFromQE"] = self.time_tracker["RemoveFromQE"] + (
                            time.time() - current_time)

                        plot_time_start = time.time()
                        # plot the new node and edge
                        cv2.circle(img, (x_m.x, x_m.y), 1, (0, 255, 0), 2)
                        cv2.line(img, (v_m.x, v_m.y),
                                 (x_m.x, x_m.y), (255, 0, 0))
                        self.total_plotting_time = self.total_plotting_time + (
                            time.time() - plot_time_start)

            else:  # Not cond1
                Q_E = []
                Q_V = []

            plot_time_start = time.time()
            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            if show_img:
                plan = self._follow_parent_pointers(dest_state)
                draw_plan(img, plan, bgr=(0, 0, 255), thickness=2,
                          image_name=imgs_path+"bitstar_result_"+str(run_num)+".png", show_img=show_img)

            self.total_plotting_time = self.total_plotting_time + \
                (time.time() - plot_time_start)

            if (step % 20 == 0) or (step == (max_num_steps-1)):
                self.current_iteration_arr.append(step)
                self.current_path_cost_arr.append(
                    dest_state.current_cost_to_come)
                self.any_path_found_arr.append(
                    dest_state.current_cost_to_come != float("Inf"))
                self.num_collision_checks_arr.append(self.num_collision_checks)
                self.batch_size_arr.append(self.prev_batch_size)
                self.cumulative_sampled_arr.append(self.cumulative_sampled)

                self.current_time_elapsed_arr.append(time.time() - start_time)
                self.time_tracker["TotalExPlotting"] = time.time(
                ) - start_time - self.total_plotting_time
                self.current_time_ex_plotting_elapsed_arr.append(
                    self.time_tracker["TotalExPlotting"])

            # Perhaps make this less restrictive?
            if dest_state.current_cost_to_come == dest_state.euclidean_distance(start_state):
                print("Optimum Solution found")
                break

        if show_img:
            cv2.waitKey(0)

        # Draw it now
        if not show_img:
            plan = self._follow_parent_pointers(dest_state)
            draw_plan(img, plan, bgr=(0, 0, 255), thickness=2,
                      image_name=imgs_path+"bitstar_result_"+str(run_num)+".png", show_img=show_img)

        return plan
