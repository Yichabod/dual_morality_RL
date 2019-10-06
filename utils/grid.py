import numpy as np
from utils import create_wall_mask, Train, create_agent_mask
import random

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
    actions = set([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])
                        

    def __init__(self, size, num_agents=1):
        assert isinstance(size, int)
        actions = set([(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)])
        
        self.size = size

        self.grid_coords = set((i,j) for i in range(size) for j in range(size))
        #self.wall_mask = create_wall_mask(shape)
        
        self.train = Train(size)
        self.other_agents = create_agent_mask()

        self._place_agent()

    def checkRep() -> None:
        """
        ensure that all states within grid are legal
        for development
        """
        pass

    def _place_agent(self) -> None:
        empty = self.grid_coords - {self.train.pos} - set(self.other_agents)
        self.agent_pos = random.choice(tuple(empty))


    def legal_actions(self) -> set:
        """
        return the list of np arrays that are legal actions
        """
        legal_actions = actions
        if self.agent_pos[0] == self.size-1:
            legal_actions-={(1,0)}
        if self.agent_pos[1] == self.size-1:
            legal_actions-={(0,1)}
        if self.agent_pos[0] == 0:
            legal_actions-={(-1,0)}
        if self.agent_pos[1] == 0:
            legal_actions-={(0,-1)}
        return legal_actions
        


    def T(self, action:tuple) -> None:
        """
        Precondition: action needs to be legal
        Returns new state, internally updates
        """
        #check that action is legal
        if action not in self.legal_actions():
            raise "Not a valid action"
        
        ag = self.agent_pos
        ac = action
        self.train.update
        new_agent_pos = (ag[0] + ac[0], ag[1] + ac[1]) 
        
        #collision detect
        if new_agent_pos == self.train.pos:
            #death
            pass
        if set(self.other_agents.keys()).intersection(new_agent_pos):
            #push
            pass
        if set(self.other_agents.keys()).intersection(self.train):
            #death
            pass
            
    
        
        #return state+action #avoid aliasing

    def R(self, state: tuple, action: tuple) -> int:
        """
        """
        
        #reward per state
    
