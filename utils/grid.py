import torch
import numpy as np
from typing import Union, Dict, Tuple

class Grid:
    '''
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions
    Takes care of transitions between states

    '''

    # available actions: stay, north, east, south, west
    actions = np.asarray([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]],
                         dtype=int)

    def __init__(self, shape: Union[int, tuple]):
        if isinstance(shape, int):
            shape = (shape, shape)
        assert isinstance(shape, tuple)
        self.shape = shape


        self.wall_mask = np.zeros(self.shape, dtype=float)
        
        self.goal_mask = np.zeros(self.shape, dtype=float)

        self._place_agent()


    def _place_agent(self) -> None:
        pass

    def T(self, action:np.ndarray, state: np.ndarray) -> tuple:
        """ Returns new state, internally updates"""
        pass
