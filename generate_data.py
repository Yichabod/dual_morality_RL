#!/usr/bin/env python3
from new_grid import Grid
import numpy as np
from agent import Agent
import time
from graphics import display_grid
import random


ELEMENT_INT_DICT = {'agent':1,'other':2,'train':3,'switch':4}
GRID_TYPE_DICT = {0:'push only',1:'switch or push',2:'do nothing',3:'others death'}

def _add_next_train_targets(grids,target1,target2):
    """
    grids: (n,5,5) np arrays where n is num actions taken
    returns: (n,2,5,5) np array with previous train position added on
    """
    shape = grids[0].shape
    ans = [] #stores full stacked grid rep.
    target_mask = np.zeros(shape,dtype=int)
    target_mask[target1[0],target1[1]] = 1
    target_mask[target2[0],target2[1]] = 2

    for ind, grid in enumerate(grids):
        if ind == len(grids)-1:
            stacked = np.stack([grid,np.zeros(shape,dtype=int),target_mask])
        else:
            next_grid = grids[ind+1]
            next_train = np.where(next_grid==ELEMENT_INT_DICT['train'], 1, 0) #zeros except for prev train pos
            stacked = np.stack([grid,next_train,target_mask])
        ans.append(stacked)
    return np.array(ans)


def get_within(steps,location):
    within = set([location])
    pos_list = [location]
    possible_moves = (1,0),(-1,0),(0,1),(0,-1)
    for i in range(steps):
        new_pos_list = []
        for pos in pos_list:
            for move in possible_moves:
                new_pos = (pos[0]+move[0],pos[1]+move[1])
                if new_pos not in within:
                    new_pos_list.append(new_pos)
                    within.add(new_pos)
        pos_list = new_pos_list
    return within

def grid_must_push(size):
    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size-2)+1 #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    train_path = set()
    for i in range(5):
        train_path.add((train_pos[0]+i*train_vel[0],train_pos[1]+i*train_vel[1]))
    open_grid_coords.remove(train_pos)

    dist = np.random.choice(3)+2
    other_1 = (train_pos[0]+train_vel[0]*dist,train_pos[1]+train_vel[1]*dist)
    open_grid_coords.remove(other_1)

    agent_pos_choices = get_within(dist-2,(other_1[0]+train_vel[1],other_1[1]+train_vel[0]))
    agent_pos_choices = agent_pos_choices.union(get_within(dist-2,(other_1[0]-train_vel[1],other_1[1]-train_vel[0])))
    agent_pos_choices = open_grid_coords.intersection(agent_pos_choices)
    agent_pos_choices -= set([(2,2)])
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    #rest of vars cannot be on train path
    open_grid_coords -= train_path

    unreachable_pos_choices = open_grid_coords-get_within(dist,agent_pos)
    other_2,switch_pos = random.sample(unreachable_pos_choices, 2)
    open_grid_coords.remove(other_2)
    open_grid_coords.remove(switch_pos)

    target_1,target_2 = random.sample(open_grid_coords, 2)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'other1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'other2':other_2,'num2':2,'target2':target_2}

def grid_must_switch(size):
    #encompasses switch only and switch or push choice cases. also doesnt exclude switch dilemma

    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size) #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    
    train_path = set()
    for i in range(5):
        train_path.add((train_pos[0]+i*train_vel[0],train_pos[1]+i*train_vel[1]))
    open_grid_coords.remove(train_pos)

    dist = np.random.choice(4)+1
    other_1 = (train_pos[0]+train_vel[0]*dist,train_pos[1]+train_vel[1]*dist)
    open_grid_coords.remove(other_1)

    open_grid_coords -= train_path
    switch_pos = random.sample(open_grid_coords,1)[0]
    open_grid_coords.remove(switch_pos)

    agent_pos_choices = open_grid_coords.intersection(get_within(dist,switch_pos))
    agent_pos_choices -= set([(2,2)])
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)


    target_1, target_2, other_2 = random.sample(open_grid_coords, 3)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'other1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'other2':other_2,'num2':2,'target2':target_2}

def grid_get_targets(size):
    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size) #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    train_path = set()
    for i in range(5):
        train_path.add((train_pos[0]+i*train_vel[0],train_pos[1]+i*train_vel[1]))

    open_grid_coords.remove(train_pos)
    open_grid_coords -= train_path

    target_1 = random.sample(open_grid_coords,1)[0]
    open_grid_coords.remove(target_1)

    corner_coords = set([(0,0),(0,4),(4,0),(4,4)])
    other_1_choices = open_grid_coords.intersection(get_within(2,target_1))-corner_coords
    other_1 = random.sample(other_1_choices,1)[0]
    open_grid_coords.remove(other_1)

    move_vector = [target_1[0]-other_1[0],target_1[1]-other_1[1]]
    if move_vector[0]>0: move_vector[0] = 1
    if move_vector[0]<0: move_vector[0] = -1
    if move_vector[1]>0: move_vector[1] = 1
    if move_vector[1]<0: move_vector[1] = -1

    agent_pos_choices = get_within(2,(other_1[0]-move_vector[0],other_1[1])).union(get_within(2,(other_1[0],other_1[1]-move_vector[1])))
    agent_pos_choices = open_grid_coords.intersection(agent_pos_choices)
    if len(agent_pos_choices) == 0:
        return False
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    open_grid_coords -= get_within(2,other_1)
    other_2, target_2, switch_pos = random.sample(open_grid_coords,3)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1


    return {'train':train_pos,'trainvel':train_vel,'other1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'other2':other_2,'num2':2,'target2':target_2}


def grid_nothing_lose(size):
    open_grid_coords = set((i,j) for i in range(size) for j in range(size))

    train_orientation = np.random.choice(4) #random between 0-3, 0->right,1->left,2->down,3->up
    train_loc = np.random.choice(size) #position along starting wall
    train_map = {0:((0,1),(train_loc,0)),1:((0,-1),(train_loc,size-1)),
                 2:((1,0),(0, train_loc)),3:((-1,0),(size-1, train_loc))}
    train_vel, train_pos = train_map[train_orientation]
    open_grid_coords.remove(train_pos)

    dist = np.random.choice(3)+1
    other_1 = (train_pos[0]+train_vel[0]*dist,train_pos[1]+train_vel[1]*dist)
    open_grid_coords.remove(other_1)

    agent_pos_choices = open_grid_coords - get_within(dist,other_1)
    agent_pos = random.sample(agent_pos_choices,1)[0]
    open_grid_coords.remove(agent_pos)

    switch_pos_choices = open_grid_coords-get_within(dist,agent_pos)
    switch_pos = random.sample(switch_pos_choices, 1)[0]
    open_grid_coords.remove(switch_pos)

    target_1,target_2,other_2 = random.sample(open_grid_coords, 3)

    if np.random.choice(2) == 1:
        other_1, other_2 = other_2, other_1
        target_1, target_2 = target_2, target_1

    return {'train':train_pos,'trainvel':train_vel,'other1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'other2':other_2,'num2':2,'target2':target_2}



def collect_grid(size, grid_type):
    """
    param: size of grid
    returns 2 ndarrays, grids_array (n, 2, size, size) and actions_value_array (n,5) generated
    by the MC agent from a single random grid
    """

    func_dict = {'push': grid_must_push,'switch':grid_must_switch,'targets':grid_get_targets,'lose':grid_nothing_lose}
    
    #PUSH includes save or save and put into target (Rewards 0,1,2)
    #SWITCH includes same as push. -1 excluded because either mc is bad or there is a dilemma. 
    #1 is excluded because may have let 1 die to get targets for 2
    #TARGETS includes 1,2,3 for successfully getting any combo of boxes into targets
    #LOSE means one box must be hit by train, possible to get 2nd into target
    valid_dict = {'push':[0,1,2],'switch':[0,2],'targets':[1,2,3],'lose':[-2,-1,1]}
    
    valid_rewards = valid_dict[grid_type]
    reward = -100 #invalid
    while reward not in valid_rewards:
        init_pos = func_dict[grid_type](size)
        while init_pos == False:
            init_pos = grid_get_targets(size)
        testgrid = Grid(size,init_pos=init_pos)

        a = Agent()
        #seems like needs 50,000 iters to solve reliably....
        Q, policy = a.mc_first_visit_control(testgrid.copy(), 1000) # Q value key is (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        grids, action_values, reward = a.run_final_policy(testgrid.copy(), Q)

    target1 = testgrid.other_agents.targets[0]
    target2 = testgrid.other_agents.targets[1]
    return _add_next_train_targets(grids,target1,target2), action_values, reward, testgrid


def data_gen(num_grids=1000,grid_size=5,distribution=None):
    """
    Saves 2 ndarrays, actions_val_array (n,5) and grids_array
    (n,2, grid_size, grid_size) generated, where the second dim is for future train pos
        each grid can generate from 2-5 data points
    Distribution is a list of 4 floats adding up to 1 representing how many of each type of
    grid to generate (with each index indicating which fraction should be of a certain type
    corresponding to GRID_TYPE_DICT)

    files should appear as "grids_data.npy" and "actions_data.npy" in the same
    directory as this script
    """
    start = time.time()
    grids_data = np.empty((1,3,grid_size,grid_size),dtype=int)
    actions_data = np.empty((1, grid_size),dtype=int)
    reward_dist = {}
 
    if distribution == None:
        distribution = {'push':25,'switch':25,'targets':25,'lose':25}

    user_testing_grids = [] #accumulates grid objects, to have playable ui

    count = 0
    for type in distribution:
        num_type = int(distribution[type]*num_grids/100)
        for i in range(num_type):
            grids,actions,reward,grid_obj = collect_grid(grid_size,type)
            user_testing_grids.append((grid_obj,reward))

            if reward not in reward_dist:
                reward_dist[reward] = 1
            else:
                reward_dist[reward] += 1

            actions_data = np.concatenate((actions_data,actions))
            grids_data = np.vstack((grids_data,grids))

            count += 1
            if count % 100 == 0:
                print("generated grid",count)

    np.save("grids_data_final_apr9",grids_data[1:])
    np.save("actions_data_final_apr9",actions_data[1:])
    print("finished in", time.time()-start)
    print("reward_dist: ", reward_dist)

    return user_testing_grids

wasd_dict = {'w':(-1,0),'a':(0,-1),'s':(1,0),'d':(0,1),' ':(0,0)}
if __name__ == "__main__":
    #num grids should always be multiple of 100
    grids = data_gen(10000, distribution={'push':23,'switch':23,'targets':39,'lose':15})

"""
    random.shuffle(grids)
    for sample in grids:
        grid = sample[0]
        best_reward = sample[1]
        display_grid(grid)
        print(grid.current_state)
        reward = 0
        while not grid.terminal_state:
             #wasd-space (space to stay in place)
            i = None
            while i not in wasd_dict:
                i = input('next step: ')
            action = wasd_dict[i]

            reward += grid.R(action)

            grid.T(action)
            display_grid(grid)
            print(grid.current_state)
        print('your reward: ', reward)
        print('best reward: ', best_reward)
        print('')
"""
