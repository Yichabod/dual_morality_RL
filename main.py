import sys
import random
from agent import Agent
from utils import Train, OtherMask, Switch, in_bounds, generate_array
from grid import Grid
from graphics import display_grid
import numpy as np





def main(size = 5):
    testgrid = Grid(5,random=True)
    agent = Agent()
    Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
    print(agent.run_final_policy(testgrid.copy(), Q,display=False))
    
main()
