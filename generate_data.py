#!/usr/bin/env python3
from grid import Grid
import numpy as np
from agent import Agent
import time

ELEMENT_INT_DICT = {'agent':1,'other':2,'train':3,'switch':4}
GRID_TYPE_DICT = {0:'push only',1:'switch or push',2:'do nothing',3:'others death'}

def _add_next_train_step(grids):
    """
    grids: (n,5,5) np arrays where n is num actions taken
    returns: (n,2,5,5) np array with previous train position added on
    """
    shape = grids[0].shape
    ans = []
    for ind, grid in enumerate(grids):
        if ind == len(grids)-1:
            stacked = np.stack((grid,np.zeros(shape,dtype=int)))
        else:
            next_grid = grids[ind+1]
            next_train = np.where(next_grid==ELEMENT_INT_DICT['train'], 1, 0) #zeros except for prev train pos
            stacked = np.stack([grid,next_train])
        ans.append(stacked)
    return np.array(ans)

def collect_grid(size, grid_type):
    """
    param: size of grid
    returns 2 ndarrays, grids_array (n, 2, size, size) and actions_value_array (n,5) generated
    by the MC agent from a single random grid
    """
    testgrid = Grid(size,random=True)
    
    """
    not yet implemented in grid
    if grid_type == 'push only':
        other in path of train, no switch present or switch too far, agent close enough to push and not die
    if grid_type == 'switch or push':
        switch within reach of agent, otherin path of train
    if grid_type == 'do nothing':
        other not in path of train
    if grid_type == 'others death':
        other in path of train, agent far from other and switch
    """
        
    a = Agent()
    Q, policy = a.mc_first_visit_control(testgrid.copy(), 1000) # Q value key is (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
    grids, action_values, reward = a.run_final_policy(testgrid.copy(), Q)
    return _add_next_train_step(grids), action_values, reward    
    

def data_gen(num_grids=1000,grid_size=5,distribution=None):
    """
    Saves 2 ndarrays, actions_val_array (n,5) and grids_array (n, size, size) generated
    by the MC agent from num_grids randomly generated grids of size grid_size
    each grid can generate from 2-5 data points
    Distribution is a list of 4 floats adding up to 1 representing how many of each type of 
    grid to generate (with each index indicating which fraction should be of a certain type
    corresponding to GRID_TYPE_DICT)

    files should appear as "grids_data.npy" and "actions_data.npy" in the same
    directory as this script
    """
    start = time.time()
    grids_data = np.empty((1,2,grid_size,grid_size),dtype=int)
    actions_data = np.empty((1, grid_size),dtype=int)
    reward_dist = {}
    for i in range(num_grids):
        if distribution == None:
            grid_type = 'random'
        else: 
            grid_type = GRID_TYPE_DICT[np.random.choice(np.arange(len(distribution)), p=distribution)]
        grids,actions,reward = collect_grid(grid_size,grid_type)
        
        if reward not in reward_dist:
            reward_dist[reward] = 1
        else:
            reward_dist[reward] += 1
        actions_data = np.concatenate((actions_data,actions))
        grids_data = np.vstack((grids_data,grids))
        if i % 100 == 0:
            print("generated grid",i)
    #print(grids_data[1:],actions_data[1:])
    np.save("grids_data",grids_data[1:])
    np.save("actions_data",actions_data[1:])
    print("finished in", time.time()-start)
    print("reward_dist: ", reward_dist)

if __name__ == "__main__":
    data_gen(10000)
