import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Config Data
with open("../config/config.json") as config_file:
    config_data = json.load(config_file)

maps_config_data = config_data['maps']

map_idxs = config_data["map_idxs"]
map_idxs = np.subtract(map_idxs, 1)

f, axarr = plt.subplots(4, 2)

x = 0
y = 0

# Maps
for i in map_idxs:
    #
    map_config_data = maps_config_data[f'map{i}']
    map_path = map_config_data["path"]

    # read the image
    im = Image.open(map_path)

    # show image
    # im.show()
    axarr[x, y].imshow(im)

    y += 1

    if y == 2:
        x += 1
        y = 0

plt.show()
