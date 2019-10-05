import numpy as np


def create_wall_mask(shape):
    """
    returns coords of outside of walls
    """
    mask = set()
    for i in range(shape):
        mask.add((0,i))
        mask.add((i,0))
        mask.add((shape-1,i))
        mask.add((i,shape-1))
    return mask
