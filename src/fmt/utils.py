import cv2
import math
import numpy as np
import pickle

from PIL import Image

class FileFormatError(Exception):
    """Exception raised for incorrect map formats.

    Attributes:
        format -- file format which caused the error
        message -- explanation of the error
    """

    def __init__(self, format, message="Incorrect file format for world map. Format should be one of ('pkl', 'png')"):
        self.format = format
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'{self.format} -> {self.message}'

def load_from_pkl(fname):
    pkl_file = open(fname, 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()
    return world

def load_from_png(fname):
    world = np.asarray(Image.open(fname).convert("RGB"))
    return world

def load_image(fname):
    format = fname.split('.')[-1]
    if format == 'pkl':
        return load_from_pkl(fname)
    elif format == 'png':
        return load_from_png(fname)
    else:
        raise FileFormatError(format)

def draw_plan(world, plan, map_idx, sidx, n_samples, pidx, hw, bgr=(255,0,0), thickness=1, mode='test', showlive=False):
    img = np.copy(world)
    for t in range(len(plan)-1):
        pt0 = (int(plan[t].v[1]), int(plan[t].v[0]))
        pt1 = (int(plan[t+1].v[1]), int(plan[t+1].v[0]))

        cv2.line(img, pt0, pt1, bgr, thickness)

    if (showlive):
        cv2.imshow('image', img)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
    cv2.imwrite(f'output/{mode}/fmt_{mode}_result_hw_{hw}_{map_idx}_state{sidx}_n_samples_{n_samples}_run{pidx}.png', img)

def compute_ndvol(R, d):
    return ((np.power((np.pi),d/2))/math.gamma((d/2)+1))*np.power(R,d)