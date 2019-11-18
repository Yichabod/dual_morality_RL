import numpy as np

ELEMENT_INT_DICT = {'agent':1,'other':2,'train':3,'switch':4}
INT_ELEMENT_DICT = {1:'◉',2:'O',3:'T',4:'S'}

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
    dims = (mdp.size,mdp.size) #tuple eg (1,11,11)
    grid = np.full(dims, 0, dtype=int) #np has nice display built in
    others_dict = mdp.other_agents.mask

    for other in others_dict:
        grid[other[0],other[1]] = ELEMENT_INT_DICT['other']

    grid[mdp.agent_pos[0],mdp.agent_pos[1]] = ELEMENT_INT_DICT['agent'] #where the agent is
    grid[mdp.switch.pos[0], mdp.switch.pos[1]] = ELEMENT_INT_DICT['switch'] #switch
    if mdp.train.on_screen == True:
        grid[mdp.train.pos[0],mdp.train.pos[1]] = ELEMENT_INT_DICT['train']

    return grid

def visualize_array(grid):
    """
    grid: 2d numpy array with integers for each element of the grid
    prints a grid with all the elements visualised
    """
    ret = np.full(grid.shape, "_", dtype=str) #np has nice display built in
    for i, row in enumerate(grid):
        for j, elt in enumerate(row):
            if elt in INT_ELEMENT_DICT:
                ret[i][j] = INT_ELEMENT_DICT[elt]
    print(ret)


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
