import numpy as np


def in_bounds(size,position:tuple) -> bool:
    """
    given positon (y,x), checks both dimensions are within the board
    helper function called by Train and Grid classses
    """
    if 0 <= position[0] < size and 0 <= position[1] < size:
        return True
    else:
        return False

def generate_array(mdp, action=None):

    """
    Takes in a Grid mdp environment and an agent, with optional action to
    show which direction next move should be in
    Generates numpy array with main agent(1), other agents(2), train(3), switch(4)
    in the grid of given dimensions.
    Intended to be able to feed into a network
    """
    dims = (1,mdp.size,mdp.size) #tuple eg (11,11)
    grid = np.full(dims, 0, dtype=int) #np has nice display built in
    others_dict = mdp.other_agents.mask

    for other in others_dict:
        grid[0,other[0],other[1]] = 2

    #if type(action) == np.ndarray:
        #next_x, next_y = action+agent.state
        #grid[next_x,next_y] = "N"

    grid[0,mdp.agent_pos[0],mdp.agent_pos[1]] = 1 #where the agent is
    grid[0,mdp.switch.pos[0], mdp.switch.pos[1]] = 4 #switch
    if mdp.train.on_screen == True:
        grid[0,mdp.train.pos[0],mdp.train.pos[1]] = 3

    return grid


class OtherMask:
    """
    Represents other agents in Grid MDP including their position and number
    """

    def __init__(self, size, positions={(1,3)}, num=1):
        self.mask = {}
        self.positions = positions
        self.size = size
        self.num = num
        for pos in positions:
            self.mask[pos] = num
    def push(self, position, action):
        num_pushed = self.mask.pop(position)
        new_pos_y = position[0] + action[0]
        new_pos_x = position[1] + action[1]
        new_pos = (new_pos_y,new_pos_x)
        self.mask[new_pos] = num_pushed
        self.positions = set(self.mask.keys())
    def copy(self):
        return OtherMask(self.size, self.positions, self.num)

class Switch:
    """
    Represents a switch to change the track of the train within the Grid MDP
    """
    def __init__(self, size, pos=(0,4)):
        self.size = size
        self.pos = pos
        self.activated = False
    
    def copy(self):
        return Switch(self.size, self.pos)

class Train:
    """
    Represents a train within the Grid MDP and its properties
    """
    def __init__(self, size, pos=(1,0),velocity=(0,1)):
        self.pos = pos
        self.velocity = velocity
        self.size = size
        self.on_screen = True
    def update(self):
        newx = self.pos[0]+self.velocity[0]
        newy = self.pos[1]+self.velocity[1]
        self.pos = (newx,newy)
        if not in_bounds(self.size,self.pos):
            self.on_screen = False
    def get_next_position(self):
        newx = self.pos[0]+self.velocity[0]
        newy = self.pos[1]+self.velocity[1]
        return (newx,newy)
    def copy(self):
        return Train(self.size, self.pos,self.velocity)
