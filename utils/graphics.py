import numpy as np

# need to modify this 

def display_grid(mdp, agent, action=None):
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

    for wall in mdp.wall:
        grid[wall[0],wall[1]] = "W"
    if type(action) == np.ndarray:
        next_x, next_y = action+agent.state
        grid[next_x,next_y] = "N"

    state_x,state_y = agent.state
    grid[agent.sta,state_y] = "◉" #where the agent is


    grid = grid.astype(str)
    pprint(grid)
    return grid
