#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 15:06:50 2019

@author: alicezhang
"""

import sys
from grid import Grid
from graphics import display_grid
import numpy as np
from agent import Agent

def collect_random_grid(size=5):
    testgrid = Grid(size,random=True)
    a = Agent()
    Q, policy = a.mc_first_visit_control(testgrid.copy(), 1000)
    a.run_final_policy(testgrid.copy(), Q)
   
def data_gen(num_grids=500):
    for i in range(num_grids):
        collect_random_grid


collect_random_grid()