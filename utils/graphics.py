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
    grid = np.full(dims, "_", dtype=str) #np has nice display built in

    for other in mdp.other_agents:
        grid[other[0],other[1]] = str(mdp.other_agents[other])


    #if type(action) == np.ndarray:
        #next_x, next_y = action+agent.state
        #grid[next_x,next_y] = "N"

    grid[mdp.agent_pos[0],mdp.agent_pos[1]] = "◉" #where the agent is
    if mdp.train.pos != None:
        grid[mdp.train.pos[0],mdp.train.pos[1]] = "T"


    grid = grid.astype(str)
    print(grid)
    return grid
