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
    dims = (1,mdp.size,mdp.size) #tuple eg (1,11,11)
    grid = np.full(dims, 0, dtype=int) #np has nice display built in
    others_dict = mdp.other_agents.mask

    for other in others_dict:
        grid[0,other[0],other[1]] = ELEMENT_INT_DICT['other']

    grid[0,mdp.agent_pos[0],mdp.agent_pos[1]] = ELEMENT_INT_DICT['agent'] #where the agent is

    grid[0,mdp.switch.pos[0], mdp.switch.pos[1]] = ELEMENT_INT_DICT['switch'] #switch

    if mdp.train.on_screen == True:
        grid[0,mdp.train.pos[0],mdp.train.pos[1]] = ELEMENT_INT_DICT['train']

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

    def __init__(self, size, positions=[(1,3)], num=1, init={}):
        self.mask = {}
        self.size = size
        self.num = num
        self.init = init
        if len(init) > 0:
            self.mask[init['other1']] = init['other1num']
            if 'other2' in init:
                self.mask[init['other2']] = init['other2num']
                self.positions = [init['other1'], init['other2']]
            else:
                self.positions = [init['other1']]
        else:
            self.positions = positions
            for pos in positions:
                self.mask[pos] = num

    def push(self, position, action):
        num_pushed = self.mask.pop(position)
        new_pos = (position[0] + action[0], position[1] + action[1])
        self.mask[new_pos] = num_pushed
        old_pos_index = self.positions.index(position)
        self.positions[old_pos_index] = new_pos

    def copy(self):
        othernum = self.num if 'other1num' not in self.init else self.init['other1num']
        new_init = {'other1':self.positions[0], 'other1num':othernum}
        if 'other2' in self.init:
            new_init['other2'] = self.positions[1]
            new_init['other2num'] = self.init['other2num']
        other = OtherMask(self.size,init=new_init)
        return other

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
    def get_next_position(self,velocity):
        newx = self.pos[0]+velocity[0]
        newy = self.pos[1]+velocity[1]
        return (newx,newy)
    def copy(self):
        return Train(self.size, self.pos,self.velocity)
