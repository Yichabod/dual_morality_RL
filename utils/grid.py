import numpy as np
from utils import create_wall_mask

"""
To Do:
 - legal actions
 - checkRep
 - support for multiple agents?

"""


class Grid:
    '''
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions
    Takes care of transitions between states

    '''

    # available actions: stay, north, east, south, west
    actions = np.asarray([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]],
                         dtype=int)

    def __init__(self, shape, num_agents=1):
        assert isinstance(shape, int)
        self.shape = shape

        self.grid_coords = set((i,j) for i in range(shape) for j in range(shape))
        self.wall_mask = create_wall_mask(shape)

        self._place_agent()

    def checkRep() -> None:
        """
        ensure that all states within grid are legal
        for development
        """
        pass

    def _place_agent(self) -> None:
        self.agent_pos = np.random.choice(self.grid_coords - self.wall_mask)


    def legal_actions(self, agent_state) -> List:
        """
        return the list of np arrays that are legal actions
        """
        pass


    def T(self, action:np.ndarray, state: np.ndarray) -> tuple:
        """
        Precondition: action needs to be legal
        Returns new state, internally updates
        """
        self.agent_pos =  state + action
        return state+action #avoid aliasing
    
