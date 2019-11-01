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
    
    
class OtherMask:
    """
    Class to maintain location and number of other agents in grid
    """
    
    def __init__(self, size, positions={(1,3)}, num=1):
        self.mask = {}
        self.positions = positions
        for pos in positions:
            self.mask[pos] = num
    def push(self, position, action):
        num_pushed = self.mask.pop(position)
        new_pos_y = position[0] + action[0]
        new_pos_x = position[1] + action[1] 
        new_pos = (new_pos_y,new_pos_x)
        self.mask[new_pos] = num_pushed
        self.positions = set(self.mask.keys())

        
class Train:
    """
    Class to represent train and its properties
    """
    def __init__(self, size, pos=(1,0),vel=(0,1)):
        self.pos = pos
        self.vel = vel
        self.size = size
        self.on_screen = True
    def update(self):
        newx = self.pos[0]+self.vel[0]
        newy = self.pos[1]+self.vel[1]
        self.pos = (newx,newy)
        if not in_bounds(self.size,self.pos):
            self.on_screen = False
    def get_next_position(self):
        newx = self.pos[0]+self.vel[0]
        newy = self.pos[1]+self.vel[1]
        return (newx,newy)
   