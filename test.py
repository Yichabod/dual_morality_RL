import unittest
import neural_net
from utils import generate_array, in_bounds
from new_grid import Grid
from agent import Agent

class TestAgent(unittest.TestCase):
    def test_model_based(self):
        testgrid = Grid(5,random=False)
        agent = Agent()
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
        _,_, reward = agent.run_final_policy(testgrid.copy(), Q,display=False)
        #no errors should be thrown


if __name__ == "__main__":
    res = unittest.main(verbosity=3, exit=False)
