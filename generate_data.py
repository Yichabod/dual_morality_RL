#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: alicezhang
"""

from grid import Grid
import numpy as np
from agent import Agent

def collect_random_grid(size=5):
    """
    returns 2 ndarrays, actions_array (n,) and grids_array (n, size, size) generated
    by the MC agent from a single random grid
    takes as input grid size
    """
    testgrid = Grid(size,random=True)
    a = Agent()
    Q, policy = a.mc_first_visit_control(testgrid.copy(), 1000)
    return a.run_final_policy(testgrid.copy(), Q)

def data_gen(num_grids=200,grid_size=5):
    """
    Saves 2 ndarrays, actions_array (n,) and grids_array (n, size, size) generated
    by the MC agent from num_grids randomly generated grids of size grid_size
    each grid can generate from 2-5 data points

    files should appear as "grids_data.npy" and "actions_data.npy" in the same
    directory as this script
    """
    grids_data = np.empty((1,grid_size,grid_size),dtype=int)
    actions_data = np.empty((1),dtype=int)
    for i in range(num_grids):
        grids,actions = collect_random_grid(grid_size)
        actions_data = np.concatenate((actions_data,actions))
        grids_data = np.vstack((grids_data,grids))
    np.save("grids_data",grids_data[1:])
    np.save("actions_data",actions_data[1:])

#uncomment below to generate data files
# data_gen()
