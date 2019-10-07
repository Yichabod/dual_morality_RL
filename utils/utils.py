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
    """
    def __init__(self, size, positions={(0,2),(2,2)}, num=1):
        self.mask = {}
        for pos in positions:
            self.mask[pos] = num
    def push(self, position, action):
        num_pushed = self.mask.pop(position)
        new_pos_y = position[0] + action[0]
        new_pos_x = position[1] + action[1] 
        new_pos = (new_pos_y,new_pos_x)
        self.mask[new_pos] = num_pushed
    def get_mask_set(self):
        return set(self.mask)
        
class Train:
    def __init__(self, size, pos=(0,0),vel=(0,1)):
        self.pos = pos
        self.vel = vel
        self.size = size
    def update(self):
        p = self.pos
        v = self.vel
        self.pos = (p[0]+v[0],p[1]+v[1])
        if not in_bounds(self.size,self.pos):
            self.pos = None
   