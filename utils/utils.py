import numpy as np


class Train:
    def __init__(self, shape, pos=(0,0),vel=(1,0)):
        self.pos = pos
        self.vel = vel
        self.shape = shape
    def update(self):
        p = self.pos
        v = self.vel
        self.pos = (p[0]+v[0],p[1]+v[1])
        if self.out_of_bounds():
            self.pos = None
            
    def out_of_bounds(self):
        if self.pos[0] > self.shape-1:
            return True
        if self.pos[1] > self.shape-1:
            return True
        if self.pos[0] < 0:
            return True
        if self.pos[1] < 0:
            return True
        return False
        
def create_agent_mask():
    return {(0,3):1}
        
     