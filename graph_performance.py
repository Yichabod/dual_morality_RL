import src.neural_net
from src.utils import generate_array, in_bounds
from src.grid import Grid
from src.agent import Agent
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

TEST_GRIDS = {
 # "1": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (3, 1), "target1": (1, 1), "switch": (0, 0), "agent": (3, 3), "cargo2": (4, 3), "target2": (1, 3), "best_reward": 1, "num1":1, "num2":2},
 # "2": {"train": (4, 3), "trainvel": (-1, 0), "cargo1": (3, 4), "target1": (4, 2), "switch": (4, 4), "agent": (1, 2), "cargo2": (1, 3), "target2": (2, 0), "best_reward": 0, "num1":1, "num2":2},
 # "3": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (3, 1), "target1": (4, 0), "switch": (4, 4), "agent": (3, 2), "cargo2": (1, 3), "target2": (1, 4), "best_reward": 2, "num1":1, "num2":2},
 # "4": {"train": (4, 0), "trainvel": (0, 1), "cargo1": (3, 2), "target1": (2, 4), "switch": (3, 0), "agent": (1, 3), "cargo2": (1, 1), "target2": (0, 1), "best_reward": 2, "num1":1, "num2":2},
 # "5": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (3, 4), "target1": (2, 4), "switch": (1, 2), "agent": (4, 3), "cargo2": (1, 0), "target2": (2, 3), "best_reward": -1, "num1":1, "num2":2},
 # "6": {"train": (0, 2), "trainvel": (1, 0), "cargo1": (4, 4), "target1": (2, 3), "switch": (1, 4), "agent": (3, 0), "cargo2": (1, 1), "target2": (1, 3), "best_reward": 2, "num1":1, "num2":2},
 # "7": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (4, 3), "target1": (3, 3), "switch": (3, 1), "agent": (0, 4), "cargo2": (1, 4), "target2": (2, 2), "best_reward": 0, "num1":1, "num2":2},
 # "8": {"train": (4, 0), "trainvel": (-1, 0), "cargo1": (2, 3), "target1": (0, 3), "switch": (4, 3), "agent": (3, 4), "cargo2": (0, 0), "target2": (3, 2), "best_reward": 1, "num1":1, "num2":2},
 # "9": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (2, 2), "target1": (4, 2), "switch": (4, 0), "agent": (1, 2), "cargo2": (4, 3), "target2": (4, 4), "best_reward": 1, "num1":1, "num2":2},
 # "10": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (3, 3), "target1": (4, 3), "switch": (2, 2), "agent": (2, 0), "cargo2": (0, 1), "target2": (0, 3), "best_reward": 2, "num1":1, "num2":2},
 # "11": {"train": (0, 0), "trainvel": (1, 0), "cargo1": (1, 0), "target1": (1, 3), "switch": (4, 0), "agent": (0, 1), "cargo2": (3, 0), "target2": (1, 4), "best_reward": -1, "num1":1, "num2":2},
 # "12": {"train": (4, 1), "trainvel": (-1, 0), "cargo1": (2, 1), "target1": (2, 2), "switch": (0, 3), "agent": (2, 0), "cargo2": (2, 3), "target2": (4, 3), "best_reward": 1, "num1":1, "num2":2},
 # "13": {"train": (4, 0), "trainvel": (-1, 0), "cargo1": (3, 2), "target1": (1, 4), "switch": (2, 3), "agent": (2, 4), "cargo2": (3, 0), "target2": (1, 1), "best_reward": 0, "num1":1, "num2":2},
 # "14": {"train": (4, 2), "trainvel": (-1, 0), "cargo1": (0, 4), "target1": (4, 3), "switch": (4, 4), "agent": (2, 1), "cargo2": (1, 2), "target2": (3, 3), "best_reward": 0, "num1":1, "num2":2},
 # "15": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (1, 3), "target1": (3, 4), "switch": (0, 1), "agent": (2, 4), "cargo2": (2, 0), "target2": (4, 2), "best_reward": 0, "num1":1, "num2":2},
 # "16": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (0, 3), "target1": (3, 2), "switch": (0, 2), "agent": (1, 1), "cargo2": (2, 0), "target2": (4, 0), "best_reward": 2, "num1":1, "num2":2},
 # "17": {"train": (3, 4), "trainvel": (0, -1), "cargo1": (0, 1), "target1": (4, 3), "switch": (4, 0), "agent": (0, 2), "cargo2": (1, 3), "target2": (1, 2), "best_reward": 2, "num1":1, "num2":2},
 # "18": {"train": (4, 0), "trainvel": (0, 1), "cargo1": (2, 1), "target1": (0, 3), "switch": (2, 0), "agent": (1, 1), "cargo2": (4, 4), "target2": (1, 3), "best_reward": 0, "num1":1, "num2":2},

 "101":{"train": (0, 3), "trainvel": (1, 0), "cargo1": (2, 2), "target1": (0, 4), "switch": (2, 4), "agent": (2, 0), "cargo2": (3, 3), "target2": (4, 4), "best_reward": -1, "num1":1, "num2":2},
 "102":{"train": (1, 0), "trainvel": (0, 1), "cargo1": (2, 2), "target1": (3, 1), "switch": (0, 4), "agent": (3, 1), "cargo2": (1, 4), "target2": (0, 3), "best_reward": -1, "num1":1, "num2":2},
 "103":{"train": (4, 4), "trainvel": (-1, 0), "cargo1": (1, 3), "target1": (3, 2), "switch": (4, 0), "agent": (1, 1), "cargo2": (0, 4), "target2": (0, 1), "best_reward": -1, "num1":1, "num2":2},
 "104":{"train": (0, 4), "trainvel": (0, -1), "cargo1": (1, 3), "target1": (3, 2), "switch": (0, 0), "agent": (2, 3), "cargo2": (0, 2), "target2": (1, 1), "best_reward": -1, "num1":1, "num2":2},
 "105":{"train": (2, 4), "trainvel": (0, -1), "cargo1": (1, 2), "target1": (3, 2), "switch": (4, 3), "agent": (0, 3), "cargo2": (2, 1), "target2": (1, 4), "best_reward": -1, "num1":1, "num2":2},
 "106":{"train": (0, 1), "trainvel": (1, 0), "cargo1": (2, 2), "target1": (3, 0), "switch": (0, 0), "agent": (2, 4), "cargo2": (3, 1), "target2": (2, 0), "best_reward": -1, "num1":1, "num2":2},
 "107":{"train": (4, 0), "trainvel": (-1, 0), "cargo1": (2, 1), "target1": (0, 0), "switch": (4, 1), "agent": (2, 3), "cargo2": (1, 0), "target2": (3, 3), "best_reward": -1, "num1":1, "num2":2},
 "108":{"train": (4, 0), "trainvel": (0, 1), "cargo1": (3, 2), "target1": (1, 4), "switch": (0, 4), "agent": (2, 1), "cargo2": (4, 4), "target2": (2, 0), "best_reward": -1, "num1":1, "num2":2},

 "201": {"train": (3, 4), "trainvel": (0, -1), "cargo1": (4, 2), "target1": (2, 4), "switch": (2, 2), "agent": (0, 3), "cargo2": (3, 1), "target2": (1, 0), "best_reward": -1, "num1":1, "num2":2},
 "202": {"train": (0, 3), "trainvel": (1, 0), "cargo1": (1, 4), "target1": (2, 4), "switch": (4, 0), "agent": (4, 2), "cargo2": (2, 3), "target2": (2, 0), "best_reward": -1, "num1":1, "num2":2},
 "203": {"train": (1, 0), "trainvel": (0, 1), "cargo1": (0, 1), "target1": (4, 3), "switch": (3, 3), "agent": (4, 4), "cargo2": (1, 2), "target2": (0, 3), "best_reward": -1, "num1":1, "num2":2},
 "204": {"train": (0, 4), "trainvel": (0, -1), "cargo1": (1, 1), "target1": (4, 1), "switch": (2, 2), "agent": (4, 4), "cargo2": (0, 0), "target2": (3, 0), "best_reward": -1, "num1":1, "num2":2},
 "205": {"train": (0, 3), "trainvel": (1, 0), "cargo1": (2, 4), "target1": (0, 2), "switch": (4, 1), "agent": (2, 0), "cargo2": (3, 3), "target2": (1, 4), "best_reward": -1, "num1":1, "num2":2},
 "206": {"train": (2, 0), "trainvel": (0, 1), "cargo1": (1, 2), "target1": (1, 1), "switch": (4, 3), "agent": (4, 0), "cargo2": (2, 3), "target2": (3, 4), "best_reward": -1, "num1":1, "num2":2},
 "207": {"train": (4, 1), "trainvel": (-1, 0), "cargo1": (3, 0), "target1": (3, 4), "switch": (0, 4), "agent": (1, 3), "cargo2": (2, 1), "target2": (2, 4), "best_reward": -1, "num1":1, "num2":2},
 "208": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (1, 3), "target1": (1, 0), "switch": (0, 0), "agent": (3, 1), "cargo2": (0, 4), "target2": (4, 1), "best_reward": -1, "num1":1, "num2":2},

 "301": {"train": (4, 2), "trainvel": (-1, 0), "cargo1": (1, 2), "target1": (1, 4), "switch": (4, 3), "agent": (3, 4), "cargo2": (1, 0), "target2": (0, 3), "best_reward": 0, "num1":1, "num2":2},
 "302": {"train": (4, 4), "trainvel": (0, -1), "cargo1": (4, 0), "target1": (2, 4), "switch": (1, 2), "agent": (0, 3), "cargo2": (1, 4), "target2": (2, 1), "best_reward": 0, "num1":1, "num2":2},
 "303": {"train": (0, 0), "trainvel": (0, 1), "cargo1": (0, 4), "target1": (1, 3), "switch": (2, 0), "agent": (4, 1), "cargo2": (4, 4), "target2": (4, 2), "best_reward": 0, "num1":1, "num2":2},
 "304": {"train": (2, 4), "trainvel": (0, -1), "cargo1": (2, 2), "target1": (0, 3), "switch": (4, 2), "agent": (4, 0), "cargo2": (0, 4), "target2": (0, 0), "best_reward": 0, "num1":1, "num2":2},
 "305": {"train": (1, 4), "trainvel": (0, -1), "cargo1": (1, 3), "target1": (4, 3), "switch": (0, 3), "agent": (0, 2), "cargo2": (2, 2), "target2": (4, 1), "best_reward": 0, "num1":1, "num2":2},
 "306": {"train": (4, 4), "trainvel": (0, -1), "cargo1": (4, 0), "target1": (0, 4), "switch": (1, 2), "agent": (0, 3), "cargo2": (1, 4), "target2": (2, 1), "best_reward": 0, "num1":1, "num2":2},
 "307": {"train": (0, 2), "trainvel": (1, 0), "cargo1": (2, 2), "target1": (3, 4), "switch": (4, 0), "agent": (3, 0), "cargo2": (4, 3), "target2": (2, 1), "best_reward": 0, "num1":1, "num2":2},
 "308": {"train": (0, 0), "trainvel": (0, 1), "cargo1": (0, 1), "target1": (3, 4), "switch": (4, 3), "agent": (4, 2), "cargo2": (1, 4), "target2": (1, 0), "best_reward": 0, "num1":1, "num2":2},

 "401": {"train": (4, 0), "trainvel": (-1, 0), "cargo1": (2, 4), "target1": (4, 4), "switch": (0, 3), "agent": (3, 3), "cargo2": (0, 1), "target2": (3, 2), "best_reward": 1, "num1":1, "num2":2},
 "402": {"train": (4, 3), "trainvel": (-1, 0), "cargo1": (3, 0), "target1": (4, 0), "switch": (0, 4), "agent": (3, 1), "cargo2": (0, 0), "target2": (1, 4), "best_reward": 1, "num1":1, "num2":2},
 "403": {"train": (4, 4), "trainvel": (-1, 0), "cargo1": (4, 2), "target1": (4, 3), "switch": (0, 0), "agent": (4, 0), "cargo2": (0, 2), "target2": (2, 0), "best_reward": 1, "num1":1, "num2":2},
 "404": {"train": (0, 3), "trainvel": (1, 0), "cargo1": (1, 4), "target1": (3, 4), "switch": (4, 0), "agent": (0, 4), "cargo2": (2, 2), "target2": (0, 1), "best_reward": 1, "num1":1, "num2":2},
 "405": {"train": (4, 2), "trainvel": (-1, 0), "cargo1": (2, 1), "target1": (2, 3), "switch": (2, 4), "agent": (4, 1), "cargo2": (1, 4), "target2": (0, 3), "best_reward": 1, "num1":1, "num2":2},
 "406": {"train": (4, 0), "trainvel": (0, 1), "cargo1": (1, 2), "target1": (0, 2), "switch": (1, 4), "agent": (3, 0), "cargo2": (2, 0), "target2": (3, 1), "best_reward": 1, "num1":1, "num2":2},
 "407": {"train": (1, 4), "trainvel": (0, -1), "cargo1": (2, 1), "target1": (2, 0), "switch": (3, 0), "agent": (2, 4), "cargo2": (0, 1), "target2": (4, 4), "best_reward": 1, "num1":1, "num2":2},
 "408": {"train": (0, 3), "trainvel": (1, 0), "cargo1": (1, 4), "target1": (2, 4), "switch": (4, 0), "agent": (0, 4), "cargo2": (0, 0), "target2": (0, 1), "best_reward": 1, "num1":1, "num2":2}
}

ITERS = [i for i in range(0,200,10)] #+ [j for j in range(175,400,25)] + [k for k in range(500,1000,100)]
REPEATS = 10 #number of times to redo the iteration; for consistency

def plot_grid_2_mc():
    test_grids = TEST_GRIDS
    all_test_list = [(key, grid) for key, grid in test_grids.items()]
    sorted(all_test_list, key=lambda x:x[0])
    agent = Agent()
    iters = ITERS
    total_normal_grid_score, total_grid1_score, total_grid2_score, total_grid3_score, total_grid4_score = [],[],[],[],[]
    repeats = REPEATS
    # for n in iters:
    #   print("Running iteration {n}".format(n=n))
    grid2_score, grid4_score = [],[]
    for ind, grid_init in all_test_list:
      normalized_score = 0
      for j in range(repeats):
          grid_num = int(ind) #ind initially is a string.
          if (grid_num < 200) or (grid_num > 300):
              continue

          best_reward = grid_init['best_reward']
          testgrid = Grid(5,random=False, init_pos=grid_init)
          if grid_num in {204, 208}:
              Q, policy = agent.mc_first_visit_control(testgrid.copy(), iters=500)
              _, _, mc_reward = agent.run_final_policy(testgrid.copy(), Q, display=True)
          else:
              continue
          normalized_score += mc_reward - best_reward
          if normalized_score != 0:
              print("Grid num {0} did not achieve best score".format(grid_num))
    #       if grid_num < 300: #grid type 2
    #           grid2_score.append(normalized_score/repeats)
    #       else: #grid type 4
    #           grid4_score.append(normalized_score/repeats)
    #   total_normal_grid_score.append(np.mean(normal_grid_score))
    #   total_grid1_score.append(np.mean(grid1_score))
    #   total_grid2_score.append(np.mean(grid2_score))
    #   total_grid3_score.append(np.mean(grid3_score))
    #   total_grid4_score.append(np.mean(grid4_score))
    # # plt.plot(iters, total_normal_grid_score, label="normal grids", color="red")
    # plt.plot(iters, total_grid1_score, label='grid 1', color="blue")
    # plt.plot(iters, total_grid2_score, label='grid 2', color="green")
    # plt.plot(iters, total_grid3_score, label='grid 3', color="orange")
    # plt.plot(iters, total_grid4_score, label='grid 4', color="brown")
    # plt.legend()
    # plt.xlabel("Number of MC Iterations")
    # plt.ylabel("Normalized Score")
    # plt.title("MC performance on all test grids")
    # plt.show()

def graph_dual_model_performance():
    test_grids = TEST_GRIDS
    all_test_list = [(key, grid) for key, grid in test_grids.items()]
    sorted(all_test_list, key=lambda x:x[0])
    agent = Agent()
    iters = ITERS
    total_normal_grid_score, total_grid1_score, total_grid2_score, total_grid3_score, total_grid4_score = [],[],[],[],[]
    repeats = REPEATS
    for n in iters:
      print("Running iteration {n}".format(n=n))
      normal_grid_score, grid1_score, grid2_score, grid3_score, grid4_score = [],[],[],[],[]
      for ind, grid_init in all_test_list:
          normalized_score = 0
          for j in range(repeats):
              grid_num = int(ind) #ind initially is a string.
              best_reward = grid_init['best_reward']
              testgrid = Grid(5,random=False, init_pos=grid_init)
              Q, policy = agent.mc_first_visit_control(testgrid.copy(), iters=n, nn_init=True)
              _, _, dual_model_reward = agent.run_final_policy(testgrid.copy(), Q, nn_init=True, display=False)
              normalized_score += dual_model_reward - best_reward
          if grid_num < 100:
              normal_grid_score.append(normalized_score/repeats)
          elif grid_num < 200: #grid type 1
              grid1_score.append(normalized_score/repeats)
          elif grid_num < 300: #grid type 2
              grid2_score.append(normalized_score/repeats)
          elif grid_num < 400: #grid type 3
              grid3_score.append(normalized_score/repeats)
          else: #grid type 4
              grid4_score.append(normalized_score/repeats)
      total_normal_grid_score.append(np.mean(normal_grid_score))
      total_grid1_score.append(np.mean(grid1_score))
      total_grid2_score.append(np.mean(grid2_score))
      total_grid3_score.append(np.mean(grid3_score))
      total_grid4_score.append(np.mean(grid4_score))
    # plt.plot(iters, total_normal_grid_score, label="normal grids", color="red")
    plt.plot(iters, total_grid1_score, label='push dilemma', color="blue")
    plt.plot(iters, total_grid2_score, label='switch dilemma', color="green")
    plt.plot(iters, total_grid3_score, label='switch save', color="orange")
    plt.plot(iters, total_grid4_score, label='push get', color="brown")
    plt.legend()
    plt.xlabel("Number of MC Iterations")
    plt.ylabel("Normalized Score")
    plt.title("Dual model performance on all test grids")
    plt.show()

def graph_mc_performance():
    test_grids = TEST_GRIDS
    all_test_list = [(key, grid) for key, grid in test_grids.items()]
    sorted(all_test_list, key=lambda x:x[0])
    agent = Agent()
    iters = ITERS
    total_normal_grid_score, total_grid1_score, total_grid2_score, total_grid3_score, total_grid4_score = [],[],[],[],[]
    repeats = REPEATS
    for n in iters:
      print("Running iteration {n}".format(n=n))
      normal_grid_score, grid1_score, grid2_score, grid3_score, grid4_score = [],[],[],[],[]
      for ind, grid_init in all_test_list:
          normalized_score = 0
          for j in range(repeats):
              grid_num = int(ind) #ind initially is a string.
              best_reward = grid_init['best_reward']
              testgrid = Grid(5,random=False, init_pos=grid_init)
              Q, policy = agent.mc_first_visit_control(testgrid.copy(), iters=n)
              _, _, dual_model_reward = agent.run_final_policy(testgrid.copy(), Q, display=False)
              normalized_score += dual_model_reward - best_reward
          if grid_num < 100:
              normal_grid_score.append(normalized_score/repeats)
          elif grid_num < 200: #grid type 1
              grid1_score.append(normalized_score/repeats)
          elif grid_num < 300: #grid type 2
              grid2_score.append(normalized_score/repeats)
          elif grid_num < 400: #grid type 3
              grid3_score.append(normalized_score/repeats)
          else: #grid type 4
              grid4_score.append(normalized_score/repeats)
      total_normal_grid_score.append(np.mean(normal_grid_score))
      total_grid1_score.append(np.mean(grid1_score))
      total_grid2_score.append(np.mean(grid2_score))
      total_grid3_score.append(np.mean(grid3_score))
      total_grid4_score.append(np.mean(grid4_score))
    # plt.plot(iters, total_normal_grid_score, label="normal grids", color="red")
    plt.plot(iters, total_grid1_score, label='grid 1', color="blue")
    plt.plot(iters, total_grid2_score, label='grid 2', color="green")
    plt.plot(iters, total_grid3_score, label='grid 3', color="orange")
    plt.plot(iters, total_grid4_score, label='grid 4', color="brown")
    plt.legend()
    plt.xlabel("Number of MC Iterations")
    plt.ylabel("Normalized Score")
    plt.title("MC performance on all test grids")
    plt.show()


if __name__ == "__main__":
    graph_dual_model_performance()
    # graph_mc_performance()
    # plot_grid_2_mc()
