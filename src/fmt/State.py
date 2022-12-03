import numpy as np

from math import sqrt

class State:
    """
    N-D state.
    """
    def __init__(self, v, parent=None):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        
        self.v = np.asarray(v) # n-d numpy array
        self.parent = parent
        self.children = []

    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and np.all(self.v == state.v)

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash(self.v)

    def euclidean_distance(self, state):
        assert (state)
        return np.linalg.norm(self.v - state.v)