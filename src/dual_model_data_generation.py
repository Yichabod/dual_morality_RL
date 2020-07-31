"""
This script uses the dual agent to generate time pressure and
time delay data for analysis.
"""
import neural_net, time, random
import pandas as pd
from agent import Agent
from graphics import display_grid
from utils import generate_array, in_bounds
import new_grid as grid

random.seed(0)

# The trick test grids are 3, 12, and 14

TEST_GRIDS = {"0": {"train": (3, 4), "trainvel": (0, -1), "cargo1": (4, 2), "target1": (2, 4), "switch": (2, 2), "agent": (0, 1), "cargo2": (3, 1), "target2": (1, 0), "best_reward": -1, "num1":1, "num2":2},
"1": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (3, 1), "target1": (1, 1), "switch": (0, 0), "agent": (3, 3), "cargo2": (4, 3), "target2": (1, 3), "best_reward": 1, "num1":1, "num2":2},
"2": {"train": (4, 3), "trainvel": (-1, 0), "cargo1": (3, 4), "target1": (4, 2), "switch": (4, 4), "agent": (1, 2), "cargo2": (1, 3), "target2": (2, 0), "best_reward": 0, "num1":1, "num2":2},
"3": {"train": (0, 3), "trainvel": (1, 0), "cargo1": (2, 2), "target1": (0, 4), "switch": (2, 4), "agent": (2, 0), "cargo2": (3, 3), "target2": (3, 4), "best_reward": 1, "num1":1, "num2":2},
"4": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (3, 1), "target1": (4, 0), "switch": (4, 4), "agent": (3, 2), "cargo2": (1, 3), "target2": (1, 4), "best_reward": 2, "num1":1, "num2":2},
"5": {"train": (4, 0), "trainvel": (0, 1), "cargo1": (3, 2), "target1": (2, 4), "switch": (3, 0), "agent": (1, 3), "cargo2": (1, 1), "target2": (0, 1), "best_reward": 2, "num1":1, "num2":2},
"6": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (3, 4), "target1": (2, 4), "switch": (1, 2), "agent": (4, 3), "cargo2": (1, 0), "target2": (2, 3), "best_reward": -1, "num1":1, "num2":2},
"7": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (1, 4), "target1": (4, 1), "switch": (4, 4), "agent": (0, 3), "cargo2": (2, 2), "target2": (2, 4), "best_reward": -2, "num1":1, "num2":2},
"8": {"train": (0, 2), "trainvel": (1, 0), "cargo1": (4, 4), "target1": (2, 3), "switch": (1, 4), "agent": (3, 0), "cargo2": (1, 1), "target2": (1, 3), "best_reward": 2, "num1":1, "num2":2},
"9": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (4, 3), "target1": (3, 3), "switch": (3, 1), "agent": (0, 4), "cargo2": (1, 4), "target2": (2, 2), "best_reward": 0, "num1":1, "num2":2},
"10": {"train": (0, 2), "trainvel": (1, 0), "cargo1": (2, 2), "target1": (2, 3), "switch": (4, 0), "agent": (2, 0), "cargo2": (1, 3), "target2": (4, 2), "best_reward": -1, "num1":1, "num2":2},
"11": {"train": (4, 1), "trainvel": (-1, 0), "cargo1": (0, 0), "target1": (4, 2), "switch": (4, 3), "agent": (1, 2), "cargo2": (1, 4), "target2": (2, 4), "best_reward": 2, "num1":1, "num2":2},
"12": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (2, 2), "target1": (3, 1), "switch": (0, 4), "agent": (3, 1), "cargo2": (1, 4), "target2": (0, 3), "best_reward": -1, "num1":1, "num2":2},
"13": {"train": (4, 0), "trainvel": (-1, 0), "cargo1": (2, 3), "target1": (0, 3), "switch": (4, 3), "agent": (3, 4), "cargo2": (0, 0), "target2": (3, 2), "best_reward": 1, "num1":1, "num2":2},
"14": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (1, 3), "target1": (3, 2), "switch": (4, 1), "agent": (0, 2), "cargo2": (0, 4), "target2": (0, 1), "best_reward": -1, "num1":1, "num2":2},
"15": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (2, 2), "target1": (4, 2), "switch": (4, 0), "agent": (1, 2), "cargo2": (4, 3), "target2": (4, 4), "best_reward": 1, "num1":1, "num2":2},
"16": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (3, 3), "target1": (4, 3), "switch": (2, 2), "agent": (2, 0), "cargo2": (0, 1), "target2": (0, 3), "best_reward": 2, "num1":1, "num2":2},
"17": {"train": (0, 4), "trainvel": (0, -1), "cargo1": (1, 1), "target1": (1, 0), "switch": (2, 1), "agent": (2, 3), "cargo2": (3, 3), "target2": (2, 2), "best_reward": 1, "num1":1, "num2":2},
"18": {"train": (0, 0), "trainvel": (1, 0), "cargo1": (1, 0), "target1": (1, 3), "switch": (4, 0), "agent": (0, 1), "cargo2": (3, 0), "target2": (1, 4), "best_reward": -1, "num1":1, "num2":2},
"19": {"train": (4, 3), "trainvel": (-1, 0), "cargo1": (2, 2), "target1": (4, 2), "switch": (4, 4), "agent": (0, 2), "cargo2": (0, 0), "target2": (3, 4), "best_reward": 1, "num1":1, "num2":2},
"20": {"train": (4, 1), "trainvel": (-1, 0), "cargo1": (2, 1), "target1": (2, 2), "switch": (0, 3), "agent": (2, 0), "cargo2": (2, 3), "target2": (4, 3), "best_reward": 1, "num1":1, "num2":2},
"21": {"train": (4, 0), "trainvel": (-1, 0), "cargo1": (3, 2), "target1": (1, 4), "switch": (2, 3), "agent": (2, 4), "cargo2": (3, 0), "target2": (1, 1), "best_reward": 0, "num1":1, "num2":2},
"22": {"train": (4, 1), "trainvel": (-1, 0), "cargo1": (2, 1), "target1": (0, 2), "switch": (4, 4), "agent": (2, 4), "cargo2": (3, 3), "target2": (2, 0), "best_reward": 0, "num1":1, "num2":2},
"23": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (3, 1), "target1": (1, 1), "switch": (0, 1), "agent": (4, 1), "cargo2": (1, 2), "target2": (1, 3), "best_reward": 1, "num1":1, "num2":2},
"24": {"train": (4, 2), "trainvel": (-1, 0), "cargo1": (0, 4), "target1": (4, 3), "switch": (4, 4), "agent": (2, 1), "cargo2": (1, 2), "target2": (3, 3), "best_reward": 0, "num1":1, "num2":2},
"25": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (1, 3), "target1": (3, 4), "switch": (0, 1), "agent": (2, 4), "cargo2": (2, 0), "target2": (4, 2), "best_reward": 0, "num1":1, "num2":2},
"26": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (0, 3), "target1": (3, 2), "switch": (0, 2), "agent": (1, 1), "cargo2": (2, 0), "target2": (4, 0), "best_reward": 2, "num1":1, "num2":2},
"27": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (0, 1), "target1": (4, 3), "switch": (3, 3), "agent": (4, 4), "cargo2": (1, 2), "target2": (0, 3), "best_reward": -1, "num1":1, "num2":2},
"28": {"train": (3, 4), "trainvel": (0, -1), "cargo1": (0, 1), "target1": (4, 3), "switch": (4, 0), "agent": (0, 2), "cargo2": (1, 3), "target2": (1, 2), "best_reward": 2, "num1":1, "num2":2},
"29": {"train": (4, 0), "trainvel": (0, 1), "cargo1": (2, 1), "target1": (0, 3), "switch": (2, 0), "agent": (1, 1), "cargo2": (4, 4), "target2": (1, 3), "best_reward": 0, "num1":1, "num2":2},
"30": {"train": (0, 2), "trainvel": (1, 0), "cargo1": (0, 3), "target1": (0, 0), "switch": (1, 1), "agent": (3, 4), "cargo2": (3, 2), "target2": (1, 3), "best_reward": 0, "num1":1, "num2":2},
"31": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (2, 3), "target1": (0, 4), "switch": (4, 3), "agent": (4, 4), "cargo2": (3, 3), "target2": (0, 1), "best_reward": 0, "num1":1, "num2":2},
"32": {"train": (0, 3), "trainvel": (1, 0), "cargo1": (1, 1), "target1": (3, 4), "switch": (0, 2), "agent": (0, 0), "cargo2": (3, 3), "target2": (4, 0), "best_reward": 0, "num1":1, "num2":2}}


TEST_GRID_LIST= [(key, grid) for key, grid in TEST_GRIDS.items()]
sorted(TEST_GRID_LIST, key=lambda x:x[0]) #make sure list is sorted in place

def generate_data(mode,num_simulations=30, save=True):
    """
    Generate the dual model data from the TEST_GRID_LIST specified above.
    Will save the generated data to a csv with columns "id", "model",
        "grid_num", "reward", and "best_reward"
    Args:
        - mode (Str): "delay" or "pressure"; whether the data generated has more
            or fewer monte carlo iterations to solve the test grids
        - num_simulations (int): how many data points to generate from the model
    Returns:
        -
    """
    agent = Agent()
    start = time.time()
    print("Starting {mode} data generation".format(mode=mode))
    model_results = [] # item e.g. {'model':'constrained','grid_num':23,'reward':3,'best_reward':3,'id':10}
    # Generate dual model "time constrained scenario"
    for i in range(num_simulations):
        if mode == "pressure":
            n_iters = random.randrange(20,30) #choose a randome integer between 20 and 30 for MC iterations
        elif mode == "delay":
            n_iters = random.randrange(220,230) #note these ranges were chosen by looking at the dual model performance graph
            # in the dual_model_data_generation.ipynb

        for ind, grid_init in TEST_GRID_LIST:
            testgrid = grid.Grid(5,random=False, init_pos=grid_init)
            Q, policy = agent.mc_first_visit_control(testgrid.copy(), iters=n_iters , nn_init=True,cutoff=0.4)
            _, _, model_reward = agent.run_final_policy(testgrid.copy(), Q, nn_init=True, display=False)
            individual_info = {} #information for this particular model instantiation
            individual_info['id'] = i
            individual_info['model'] = mode
            individual_info['grid_num'] = ind
            individual_info['reward'] = model_reward
            individual_info['best_reward'] = grid_init['best_reward']
            model_results.append(individual_info)
        print("Simulation {num} took {time} seconds".format(num=i, time=time.time()-start))
        start = time.time()

    return model_results


if __name__ == "__main__":
    pressure_results = generate_data("pressure")
    delay_results = generate_data("delay")
    model_results = pressure_results+delay_results
    results_df = pd.DataFrame(model_results)
    results_df.to_csv('dual_model_data_generation.csv')
