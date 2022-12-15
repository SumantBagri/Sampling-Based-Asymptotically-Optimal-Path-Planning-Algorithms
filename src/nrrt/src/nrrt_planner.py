#!/usr/bin/env python3
import random
import time
import re
import numpy as np
import pandas as pd
import cv2
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from PIL.Image import fromarray
from cnn.nrrt_cnn import NRRTCNN
from plotting_utils import draw_plan

MODEL_PATH = "../model/model_007_050.pth"
CONFIG_PATH = '../config/config.json'
RESULTS_PATH = "nrrt_results.csv"

class State:
    def __init__(self, x, y, parent):
        self.x = x
        self.y = y
        self.parent = parent
        self.cost = 0
        self.children = []

    def __eq__(self, state):
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
        return hash((self.x, self.y))

    def euclidean_distance(self, state):
        assert (state)
        return np.sqrt((state.x - self.x)**2 + (state.y - self.y)**2)

    def __repr__(self):
        return "[{}, {}]".format(self.x, self.y)

class NRRTPlanner:
    def __init__(self, model, img_size, dev, alpha=0.5, clearance=1):
        self.model = model
        self.img_size = img_size
        self.dev = dev
        self.alpha = 0.5
        self.clearance = 1
    
    def state_is_free(self, state):
        '''
            Collision check for state with self.clearance
        '''
        return (self.img[state.y-self.clearance:state.y+self.clearance, state.x-self.clearance:state.x+self.clearance] == 0).all()
    
    def convert_img(self, start, end):
        '''
            encodes map into cnn trainable format
            and converts to torch.tensor 
        '''
        img = np.copy(self.img)
        img[start.y, start.x] = 2
        img[end.y, end.y] = 3
        img = img.astype(dtype='float32')

        # torch conversion
        img = img.reshape(1, 1, self.img_size, self.img_size)
        img = torch.from_numpy(img).to(self.dev)
        
        return img

    def process_img(self, img):
        '''
            resizes images using bilinear interpolation
        '''
        # resize/assign values
        if img.shape != (self.img_size, self.img_size):
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = np.where(img !=255, 1, 0)
        
        return img

    def update_img(self, img):
        '''
            updates img of object with a new img
        '''
        self.world = img
        self.img = self.process_img(img)

    def update_start_goal_states(self, start, end):
        '''
            encodes the input image according to 
            the start/end states, computes
            probability distribution of map 
        '''
        self.img = self.convert_img(start, end)
        self.prob = self.prob_map(start, end)

    def prob_map(self, start, end):
        '''
            generates probability map
            based on given start, end states
        '''
        
        assert(self.state_is_free(start))
        assert(self.state_is_free(end))

        # generate prob predictions
        out = self.model(self.img)
        out = out.cpu().detach().numpy().squeeze()
        
        # threshold distribution by 0.5
        out = np.where(out < 0.5, 0, out)

        return out

    def display_prob_map(self):
        '''
            displays the probability distribution
            of states as well as the original
            map image
        '''
        fig, axes = plt.subplots(1, 2, sharey=True)

        # probability distribution
        ax1 = sns.heatmap(self.prob, ax=axes[0], cmap=sns.color_palette("Greys", as_cmap=True), cbar=False, xticklabels=False, yticklabels=False)
        ax1.tick_params(left=False, bottom=False)

        # original map
        ax1 = sns.heatmap(self.world, ax=axes[1], cmap=sns.color_palette("Greys_r", as_cmap=True), cbar=False, xticklabels=False, yticklabels=False)
        ax1.tick_params(left=False, bottom=False)

        plt.show()
    
    def uniform_sample(self, sample_size):
        '''
            uniform sample over all states in
            a given map
        '''
        xs = np.random.randint(0, self.world.shape[1], sample_size)
        ys = np.random.randint(0, self.world.shape[0], sample_size)

        return [State(b, a, None) for b, a in zip(ys, xs)]
    
    def non_uniform_sample(self, sample_size):
        '''
            uses the non-uniform distribution learned
            by the cnn to sample states
        '''
        p = softmax(self.prob)
        idx = np.random.choice(range(self.img_size * self.img_size), p=p.flatten(), size=sample_size)
        ys, xs = np.unravel_index(idx, (self.img_size, self.img_size))

        return [State(b, a, None) for b, a in zip(ys, xs)]

    def _follow_parent_pointers(self, state):
        '''
            builds path from following parent pointers
            recursively
        '''
        curr_ptr = state
        path = [state]
        cost = 0

        while curr_ptr is not None:
            path.append(curr_ptr)
            cost += curr_ptr.euclidean_distance(curr_ptr.parent)
            curr_ptr = curr_ptr.parent

        return path[::-1], cost

    def find_closest_state(self, tree_nodes, state):
        '''
            finds closest available state within
            visited nodes
        '''
        min_dist = float("Inf")
        closest_state = None

        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state

    def find_state_within_distance(self, s_start, s_dest, distance):
        '''
            finds state from s_start to the direction of
            s_dest, with a max travel distance
        '''
        # identical states
        if (s_start.x == s_dest.x and s_start.y == s_dest.y):
            return 0, 0

        # find vector from s_start to s_dest
        vec_x = s_dest.x - s_start.x
        vec_y = s_dest.y - s_start.y

        # normalize vector
        vec_x = vec_x / np.sqrt(vec_x ** 2 + vec_y **2)
        vec_y = vec_y / np.sqrt(vec_x ** 2 + vec_y **2)

        # find point distance away from s_start in direction of s_dest
        x = round(s_start.x + distance * vec_x)
        y = round(s_start.y + distance * vec_y)

        # clamp x, y values to [0, max_dim - 1]
        x = max(0, min(x, self.img.shape[1] - 1))
        y = max(0, min(y, self.img.shape[0] - 1))

        return x, y

    def steer_towards(self, s_nearest, s_rand, max_radius):
        '''
            steers robot from s_nearest to s_rand using a
            maximum steering radius
        '''
        x = 0
        y = 0

        if s_nearest.euclidean_distance(s_rand) > max_radius:
            x, y = self.find_state_within_distance(s_nearest, s_rand, max_radius)
        else:
            x = s_rand.x
            y = s_rand.y

        s_new = State(x, y, s_nearest)

        return s_new
    
    def path_is_obstacle_free(self, s_from, s_to):
        '''
            collision check between s_from and s_to
            with max_checks increments
        '''
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10

        for i in range(max_checks):
            x, y = self.find_state_within_distance(s_from, s_to, (float(i) / max_checks) * s_from.euclidean_distance(s_to))
            check_state = State(x, y, None)
            
            if not self.state_is_free(check_state):
                return False
        
        return True
    
    def plan(self, start, end, max_iterations, max_steering_radius, tolerance, sample_size=5):
        '''
            plans a path between start and end states
            using NRRT* planner
        '''
        assert (self.state_is_free(start))
        assert (self.state_is_free(end))

        # The set containing the nodes of the tree
        tree_nodes = set()
        tree_nodes.add(start)

        plan = [start]
        num_collision_checks = 0

        for step in range(max_iterations):
            # alpha % chance of non uniform sampling, (1 - alpha) % chance of uniform sampling
            samples = self.uniform_sample(sample_size) if random.random() < self.alpha else self.non_uniform_sample(sample_size)

            for s_rand in samples:
                s_nearest = self.find_closest_state(tree_nodes, s_rand)
                s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)
                num_collision_checks += 1

                if self.path_is_obstacle_free(s_nearest, s_new):
                    tree_nodes.add(s_new)
                    s_nearest.children.append(s_new)
                                    
                    if s_new.euclidean_distance(end) < tolerance:
                        end.parent = s_new
                        plan, cost = self._follow_parent_pointers(end)
                        break
            # ran out of samples - re sample
            else:
                continue
            # path found
            break
        # ran out of iterations - failed
        else:
            plan = [start]

        return plan, cost, step, num_collision_checks

    def plot_tree(self, plan):
        # image to be used to display the tree
        img = np.copy(self.world)

        for node in self.complete_tree:
            cv2.circle(img, (node.x, node.y), 2, (0, 0, 0))
            
            if node.parent is not None:
                cv2.line(img, (node.parent.x, node.parent.y), (node.x, node.y), (255, 0, 0), thickness=1)
                
        draw_plan(img, plan, bgr=(0,0,255), thickness=2)
        cv2.waitKey(0)

def simulation(planner, start, end, max_num_steps, max_steering_radius, batch_size, check_div):
    try:
        start_time = time.time()
        plan, cost, num_iterations, num_collision_checks = planner.plan(start, end, max_num_steps, max_steering_radius, max_steering_radius, batch_size)
        end_time = time.time() - start_time                 
    except Exception as e:
        print(repr(e)):
        cost = -1
        num_iterations = -1
        end_time = -1
        num_collision_checks = -1
        
    return cost, num_iterations, end_time, num_collision_checks

if __name__ == "__main__":
    # load model
    dev = torch.device('cuda')
    chckpt = torch.load(MODEL_PATH)
    model = NRRTCNN(1, 1).to(dev)
    model.load_state_dict(chckpt['model'])

    # load config file
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # simulation config
    seeds = config['rng_seeds']
    max_iterations = config['max_num_steps']
    map_idxs = np.array(config['map_idxs']) - 1
    num_state_pairs = config['state_pair_per_map'] 
    num_runs = config['runs_per_map']
    random_seeds = config['rng_seeds']

    # nrrt specific config
    max_check_div = config['nrrt']['max_div']
    batch_sizes = config['nrrt']['batch_size']
    ovr_idx = config['nrrt']['start_idx']
    max_steering_radius = config['nrrt']['steering_radius']
    total_num_simulations = len(map_idxs) * num_runs * num_state_pairs * len(batch_sizes)
    
    planner = NRRTPlanner(model, 256, dev)
    result = {"Overall Test Number": list(range(ovr_idx, ovr_idx + total_num_simulations)),
            "Algorithm": ['NRRT*'] * total_num_simulations,
            "Map Type": [],
            "Map Id": [],
            "Start Point": [],
            "Goal Point": [],
            "Test Number": list(range(5)) * (total_num_simulations // 5),
            "Iteration": [],
            "Timestep": [],
            "Num Collision Checks": [],
            "Batch Size": [],
            "Cumulative Num Sampled": [],
            "Current Path Cost": [],
            "Any Path Found": []}

    map_keys = np.array([*config['maps'].keys()])[map_idxs]
    map_entries = [config['maps'][k] for k in map_keys]

    for m in map_entries:
        path = config['maps'][m]['path']

        map_name = m['path'].split("/")[-1]
        map_id = re.findall(r'\d+', map_name)[0]
        map_type = re.findall(r'\d+_(.*?)\.', map_name)[0]

        # prep data for planning
        img = cv2.imread(path.split("../")[1])
        planner.update_img(img)
        
        # start/target states 
        start = [State(m[x]['start'][0], m[x]['start'][1], None) 
                 for x in m.keys() if x != 'path']
        end = [State(m[x]['target'][0], m[x]['target'][1], None) 
               for x in m.keys() if x != 'path']

        for b_size in batch_sizes:
            for s, e in zip(start, end):
                for i in range(5):    
                    planner.update_start_goal_states(s, e)
                    cost, num_iterations, time, num_collision_checks = simulation(planner, s, e, max_iterations, max_steering_radius, b_size, max_check_div)
                    
                    result['Map Type'].append(map_type)
                    result['Map Id'].append(map_id)
                    result['Start Point'].append(s)
                    result['Goal Point'].append(e)
                    result['Timestep'].append(end_time)
                    result['Iteration'].append(num_iterations)
                    result['Num Collision Checks'].append(num_collision_checks)
                    result['Batch Size'].append(b_size)
                    result['Cumulative Num Sampled'].append(num_iterations)
                    result['Current Path Cost'].append(cost)
                    result['Any Path Found'].append(True if cost > 0 else False)
                    
    df = pd.DataFrame(result)
    df.to_csv(RESULTS_PATH)
