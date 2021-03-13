import unittest
import src.neural_net
from src.utils import generate_array, in_bounds
from src.grid import Grid
from src.agent import Agent
import matplotlib.pyplot as plt
import numpy as np

class TestAgent(unittest.TestCase):



    def test_models_run(self):
        testgrid = {"train": (2, 0), "trainvel": (0, 1), "cargo1": (3, 1), "target1": (1, 1), "switch": (0, 0), "agent": (3, 3), "cargo2": (4, 3), "target2": (1, 3), "best_reward": 1, "num1":1, "num2":2}
        testgrid = Grid(5,random=False, init_pos=testgrid)
        agent = Agent()
        #model based
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
        _,_, reward = agent.run_final_policy(testgrid.copy(), Q,display=False)
        #model free
        agent.run_model_free_policy(testgrid.copy(), display=True)
        #dual model
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 10, nn_init=True)
        agent.run_final_policy(testgrid.copy(), Q,nn_init=True,display=True)
        #no errors should be thrown


if __name__ == "__main__":
    res = unittest.main(verbosity=3, exit=False)
