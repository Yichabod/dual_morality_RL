import numpy as np
import string

def display_grid(mdp, action=None):

    """
    Takes ina Grid mdp environment and an agent, with optional action to
    show which direction next move should be in
    Displays main agent(◉), other agents(numbers), switch(S) and train (V,<,>,^),
    targets ('a' for cargo 1, 'b' for cargo 2)
    in the grid of given dimensions
    Call the displayGrid function right after next action is generated
    but before it moves
    """
    dims = (mdp.size,mdp.size) #tuple eg (11,11)
    grid = np.full(dims, "_", dtype=str) #np has nice display built in
    others_dict = mdp.other_agents.get_mask()
    others_list = sorted(list(others_dict.values()), key = lambda x: x.num) #sorted lowest num first

    #target for 1 displayed as a, target for 2 displayed as b
    target_dict = {}
    for ind, other in enumerate(others_list):
        target_dict[other.num] = string.ascii_lowercase[ind]

    # target_dict = {others_list[0]:'a',others_list[1]:'b'}

    velocity_dict = {(1,0):'v',(0,1):'>',(-1,0):'^',(0,-1):'<'}

    grid[mdp.switch.pos[0], mdp.switch.pos[1]] = "S"

    for other in others_dict:
        num = others_dict[other].get_num()
        target_pos = others_dict[other].get_target()
        grid[target_pos[0],target_pos[1]] = target_dict[num]
        grid[other[0],other[1]] = str(num)

    grid[mdp.agent_pos[0],mdp.agent_pos[1]] = "◉" #where the agent is
    if mdp.train.on_screen == True:
        # if agent is killed by train, X marks collision
        if mdp.train.pos == mdp.agent_pos:
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "X"
        # if other is killed by train, X marks collision
        elif set(others_dict.keys()).intersection({mdp.train.pos}):
            grid[mdp.train.pos[0],mdp.train.pos[1]] = "x"
        else:
            train_velocity = mdp.train.velocity
            grid[mdp.train.pos[0],mdp.train.pos[1]] = velocity_dict[train_velocity]

    print(grid)
    print("="*len(grid))
    return grid
