from math import sqrt


class State:
    """
    2D state.
    """

    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []
        self.current_cost_to_come = float("Inf")

    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and self.x == state.x and self.y == state.y

    # Added for heapq
    def __lt__(self, state):
        """
        When is this state less than another state?
        """
        if self.current_cost_to_come > state.current_cost_to_come:
            return True

        return False

    # Added for heapq
    def __le__(self, state):
        """
        When is this state less than or equal to another state?
        """
        if self.current_cost_to_come >= state.current_cost_to_come:
            return True

        return False

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))

    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x)**2 + (state.y - self.y)**2)
