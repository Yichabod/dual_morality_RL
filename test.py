import unittest
import src.neural_net
from src.utils import generate_array, in_bounds
from src.grid import Grid
from src.agent import Agent

class TestAgent(unittest.TestCase):
    def test_models_run(self):
        testgrid = Grid(5,random=False)
        agent = Agent()
        #model based
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
        _,_, reward = agent.run_final_policy(testgrid.copy(), Q,display=False)
        #model free
        agent.run_model_free_policy(testgrid.copy(), display=True)
        #dual model
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1, nn_init=True,cutoff=0.4)
        agent.run_final_policy(testgrid.copy(), Q,nn_init=True,display=True)
        #no errors should be thrown


if __name__ == "__main__":
    res = unittest.main(verbosity=3, exit=False)
