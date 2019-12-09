import unittest
import neural_net
from utils import generate_array, in_bounds
from grid import Grid
from agent import Agent

class TestAgent(unittest.TestCase):
    def test_model_free(self):
        testgrid = Grid(5,random=False)
        agent = Agent()
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
        _,_, reward = agent.run_final_policy(testgrid.copy(), Q,display=False)
        self.assertEqual(reward,-0.2) #should hit the switch


if __name__ == "__main__":
    res = unittest.main(verbosity=3, exit=False)
