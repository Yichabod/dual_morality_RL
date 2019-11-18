import sys
import random
from agent import Agent
from utils import Train, OtherMask, Switch, in_bounds, generate_array
from grid import Grid
from graphics import display_grid
import neural_net
import numpy as np


action_dict = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4}


def main(size = 5):
    testgrid = Grid(5,random=True)
    agent = Agent()
    Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
    print(agent.run_final_policy(testgrid.copy(), Q,display=False))


testgrid = Grid(5,random=False)
input_array = generate_array(testgrid)
neural_net.predict(net, input_array)
