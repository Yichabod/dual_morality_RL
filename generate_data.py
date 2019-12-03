#!/usr/bin/env python3
from grid import Grid
import numpy as np
from agent import Agent
import time

ELEMENT_INT_DICT = {'agent':1,'other':2,'train':3,'switch':4}

def _add_previous_train_step(grids):
    """
    grids: (n,5,5) np arrays where n is num actions taken
    returns: (n,2,5,5) np array with previous train position added on
    """
    shape = grids[0].shape
    ans = []
    for ind, grid in enumerate(grids):
        if ind == 0:
            stacked = np.stack((np.zeros(shape,dtype=int), grid))
        else:
            prev_grid = grids[ind-1]
            prev_train = np.where(prev_grid==ELEMENT_INT_DICT['train'], prev_grid,0) #zeros except for prev train pos
            stacked = np.stack([prev_train,grid])
        ans.append(stacked)
    return np.array(ans)

def collect_random_grid(size=5):
    """
    param: size of grid
    returns 2 ndarrays, grids_array (n, 2, size, size) and actions_value_array (n,5) generated
    by the MC agent from a single random grid
    """
    testgrid = Grid(size,random=True)
    a = Agent()
    Q, policy = a.mc_first_visit_control(testgrid.copy(), 1000) # Q value key is (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
    grids, action_values, _type = a.run_final_policy(testgrid.copy(), Q)
    return _add_previous_train_step(grids), action_values


def data_gen(num_grids=1000,grid_size=5):
    """
    Saves 2 ndarrays, actions_val_array (n,5) and grids_array (n, size, size) generated
    by the MC agent from num_grids randomly generated grids of size grid_size
    each grid can generate from 2-5 data points

    files should appear as "grids_data.npy" and "actions_data.npy" in the same
    directory as this script
    """
    start = time.time()
    grids_data = np.empty((1,2,grid_size,grid_size),dtype=int)
    actions_data = np.empty((1, grid_size),dtype=int)
    for i in range(num_grids):
        grids, actions = collect_random_grid(grid_size)
        actions_data = np.concatenate((actions_data,actions))
        grids_data = np.vstack((grids_data,grids))
        if i % 100 == 0:
            print("generated grid",i)
    np.save("grids_data",grids_data[1:])
    np.save("actions_data",actions_data[1:])
    print("finished in", time.time()-start)


if __name__ == "__main__":
    data_gen(10000)
