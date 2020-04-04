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

def generate_array(mdp,action=None):

    """
    Takes in a Grid mdp environment and an agent, with optional action to
    show which direction next move should be in
    Generates numpy array with main agent(1), other agents(depends on value), train(3),
    switch(4), targets(depends on value of other agent)
    in the grid of given dimensions.
    Intended to be able to feed into a network
    """
    # TODO:
    # Change this function to output 1xCx5x5 instead to allow different numbers of other agents and change
    # neural_net.py accordingly
    #
    dims = (1,mdp.size,mdp.size) #tuple eg (1,11,11)
    grid = np.full(dims, 0, dtype=int) #np has nice display built in
    others_dict = mdp.other_agents.mask

    for other_coord, other_obj in others_dict.items():
        #the value of the other in the grid will be the num of
        # non other elements + value of other
        target = other_obj.target
        grid[0,other_coord[0],other_coord[1]] = len(ELEMENT_INT_DICT)+other_obj.num
        grid[0, target[0], target[1]] = len(ELEMENT_INT_DICT)+other_obj.num + 1

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

    def __init__(self, positions=[(1,3)], num=[1], init={}, targets=[(1,4)]):
        """
        """
        self.mask = {}
        self.init = init
        self.targets = []
        self.positions = []

        if len(init) > 0:
            self.mask = init
            self.positions = list(self.mask.keys())
            for pos in self.mask:
                self.targets.append(self.mask[pos].get_target())
        else:
            self.positions = positions
            for idx,pos in enumerate(positions):
                self.mask[pos] = Other(num[idx],targets[idx],targets[idx]==pos)
            self.targets = targets

    def push(self, position, action):
        other_pushed = self.mask.pop(position,None)
        new_pos = (position[0] + action[0], position[1] + action[1])
        self.mask[new_pos] = other_pushed
        old_pos_index = self.positions.index(position)
        self.positions[old_pos_index] = new_pos

    def copy(self):
        new_init = {}
        for pos in self.mask:
            new_init[pos] = self.mask[pos].copy()
        othermask = OtherMask(init=new_init)
        return othermask

    def get_mask(self):
        return self.mask

class Other:
    def __init__(self,num,target,active):
        self.target = target
        self.num = num
        self.active = active
    def copy(self):
        return Other(self.num,self.target,self.active)
    def get_num(self):
        return self.num
    def get_target(self):
        return self.target
    def toggle_active(self):
        self.active = not self.active


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
