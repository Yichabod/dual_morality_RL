import random
import numpy as np


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

    def __init__(self, size, num_agents=1):
        assert isinstance(size, int)
        self.size = size
        self.grid_coords = set((i,j) for i in range(size) for j in range(size))
        self._place_agent()

    def checkRep() -> None:
        """
        ensure that all states within grid are legal
        for development
        """
        pass

    def _place_agent(self) -> None:
        starting_pos = random.choice(list(self.grid_coords))
        self.agent_pos = np.array(starting_pos)

    def legal_actions(self, agent_state) -> list:
        """
        return the list of np arrays that are legal actions
        """
        assert isinstance(agent_state, np.ndarray)
        legal_actions = []
        for action in self.actions:
            print("action",action)
            new_pos = action + self.agent_pos
            x_valid = new_pos[0] >= 0 and new_pos[0] < self.size #assumes square grid
            y_valid = new_pos[1] >= 0 and new_pos[1] < self.size
            if x_valid and y_valid:
                legal_actions.append(action)

        return  legal_actions


    def T(self, action:np.ndarray, state: np.ndarray) -> tuple:
        """
        Precondition: action needs to be legal
        Returns new state, internally updates
        """
        self.agent_pos =  state + action
        return state+action #avoid aliasing

if __name__ == "__main__":
    grid = Grid(4)
    assert(len(grid.legal_actions(np.array([1,1])))==4)
