import sys
# sys.path.append('/Users/maxlangenkamp/Desktop/UROP/dual_morality_RL/datagen/utils')
import random
from utils import Train, OtherMask, Switch, in_bounds, generate_array
from grid import Grid
from graphics import display_grid
import numpy as np





def main(size = 5):
    grid = Grid(size,random=True)
    grid.agent_pos
    display_grid(grid)
    print(grid.current_state)
    while not grid.terminal_state:
        print("")
        action =  tuple(grid.legal_actions())[0]#random.choice(tuple(grid.legal_actions()))
        grid.T(action)
        display_grid(grid)
        print(grid.current_state)

main()
