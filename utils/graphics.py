import numpy as np


def display_grid(mdp, action=None):
    """
    Takes in mdp environment and an agent, with optional action to
    show which direction next move should be in
    Displays walls(W), goals (G), main agent(◉), other agents(A), next states (N)
    in the grid of given dimensions
    Call the displayGrid function right after next action is generated
    but before it moves
    To Do: add support for collision
    """
    dims = (mdp.size,mdp.size) #tuple eg (11,11)
    grid = np.full(dims, " ", dtype=str) #np has nice display built in

    if type(action) == np.ndarray:
        next_x, next_y = action+agent.state
        grid[next_x,next_y] = "N"

    state_x,state_y = mdp.agent_pos
    grid[state_x,state_y] = "◉" #where the agent is


    grid = grid.astype(str)
    print(grid)
    return grid
