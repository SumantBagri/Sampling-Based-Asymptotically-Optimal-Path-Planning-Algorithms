import cv2
import numpy as np

from FMTPlanner import FMTPlanner
from State import State
from utils import load_image

rgbworld = load_image('Maps/map.png')

planner = FMTPlanner(world=rgbworld, n_samples=1000)

plan_info = planner.plan([538, 60], [60, 1130])

# start = State([180, 20])
# target = State([20, 380])

# world = (rgbworld[:,:,0] == 0).astype(np.uint8)
# prng = np.random.RandomState(0)
# node_list = list()
# while len(node_list) < 1000:
#     # sample a d-dimensional State from a uniform distribution
#     sample = State(prng.uniform(0, world.shape).astype(np.uint8))
#     node_list.append(sample)

# img = np.copy(world)
# # Plot start and target
# cv2.circle(img, (start.v[1], start.v[0]), 6, (65, 200, 245), -1)
# cv2.imshow('image', img)
# cv2.waitKey(100)
# cv2.circle(img, (target.v[1], target.v[0]), 6, (65, 245, 100), -1)
# cv2.imshow('image', img)
# cv2.waitKey(100)
# # Plot the sampled points
# for i in range(1,len(node_list)-1):
#     sample = node_list[i]
#     cv2.circle(img, (sample.v[1], sample.v[0]), 2, (245, 95, 65))
#     cv2.imshow('image', img)
#     cv2.waitKey(10)