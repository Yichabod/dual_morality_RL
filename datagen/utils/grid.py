import random
from utils import Train, OtherMask, in_bounds
from graphics import display_grid
import numpy as np


"""
To Do:
 - how to integrate reward function with transition?
 - checkRep
 - support for multiple agents?

"""


class Grid:
    '''
    Grid is the state class for the MDP, with transition and reward functions
    Provides base grid environment with which the agents interact.
    Keeps track of agent positions, others positions, and train 
    '''


    def __init__(self, size, num_agents=1):
        assert isinstance(size, int)

        # available actions: stay, north, east, south, west
        # coords are (y,x), with negative being up and left
        self.all_actions =[(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]

        self.size = size
        
        #all possible coordinates as a y,x tuple
        self.grid_coords = set((i,j) for i in range(size) for j in range(size))

        self.train = Train(size)
        self.other_agents = OtherMask(size)

        self._place_agent()
        
        #timestep horizon until end of episode
        """
        self.terminal_time = 10
        self.timestep = 0
        """
        self.terminal_state = False
        self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])


    def _place_agent(self) -> None:
        """
        randomly sets self.agent_pos taking into account currently occupied coordinates
        """
        empty = self.grid_coords - {self.train.pos} - self.other_agents.positions

        self.agent_pos = random.choice(tuple(empty))
        #for first pass MC solution
        self.agent_pos = (0,3)


    def legal_actions(self) -> set:
        """
        return the set of tuples representing legal actions from the current state
        """
        legal_actions = self.all_actions.copy()
        for action in self.all_actions:
            new_position_y = self.agent_pos[0]+action[0]
            new_position_x = self.agent_pos[1]+action[1]
            if not in_bounds(self.size,(new_position_y,new_position_x)):
                legal_actions.remove(action)

        return legal_actions


    def T(self, action:tuple) -> None:
        """
        Precondition: action needs to be legal, board cannot be in terminal state
        Returns None at the moment, changes grid object properties to update state
        can be changed return duplicate grid object if mutation is bad 
        """

        #check not terminal state
        if self.terminal_state:
            return self.agent_pos,self.train.pos
        
        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y) 
        self.train.update()
        
        #episode ends if train leaves screen or collides
        if not self.train.on_screen:
            self.terminal_state = True
        
        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos
        
        #collision detect
        if new_agent_pos == self.train.pos:
            #agent intersect train: death, terminal state
            self.train.vel = (0,0)
            self.terminal_state = True
            pass
        if self.other_agents.positions.intersection({new_agent_pos}):
            #agent intersect other: push
            #moves both agent and other given that it will not push anyone out of bounds
            new_other_y = new_agent_pos[0] + action[0]
            new_other_x = new_agent_pos[1] + action[1]
            new_other_pos = (new_other_y,new_other_x)
            if in_bounds(self.size,new_other_pos):
                self.other_agents.push(new_agent_pos,action)
            else:
                new_agent_pos = self.agent_pos
        if self.other_agents.positions.intersection({self.train.pos}):
            #other intersect train: death, terminal state
            self.train.vel = (0,0)
            self.terminal_state = True
            pass
            
        self.agent_pos = new_agent_pos
        self.current_state = (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        return self.current_state

    def R(self, action: tuple) -> int:
        """
        """
        if self.terminal_state:
            return 0 
        
        reward = 0
        
        #check that action is legal
        new_x = self.agent_pos[0] + action[0]
        new_y = self.agent_pos[1] + action[1]
        new_agent_pos = (new_x,new_y) 
        new_train_pos = self.train.get_next_position()
            
        if action not in self.legal_actions():
            new_agent_pos = self.agent_pos
            
        if new_agent_pos == new_train_pos:
            #agent intersect train: death
            reward -= 10
            
        if self.other_agents.positions.intersection({new_agent_pos}):
            #agent intersect other: push
            #moves both agent and other given that it will not push anyone out of bounds
            reward -= 1
        if self.other_agents.positions.intersection({new_train_pos}):
            #other intersect train: death, terminal state
            reward -= 4
       
        return reward


if __name__ == "__main__":
    # makes 5x5 test grid and chooses random action until terminal state is reached
    grid = Grid(5)
    display_grid(grid)
    print(grid.current_state)
    while not grid.terminal_state:
        print("")
        action = random.choice(tuple(grid.legal_actions()))
        print(action)
        print(grid.R(action))
        grid.T(action)
        display_grid(grid)
        print(grid.current_state)
        

