#!/usr/bin/env python3
from new_grid import Grid
import numpy as np
from agent import Agent
import time
from graphics import display_grid
import random
import json


ELEMENT_INT_DICT = {'agent':1,'train':2,'switch':3}
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

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}

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

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}

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


    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}


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

    return {'train':train_pos,'trainvel':train_vel,'cargo1':other_1,'num1':1,'target1':target_1,
            'switch':switch_pos,'agent':agent_pos,'cargo2':other_2,'num2':2,'target2':target_2}



def collect_grid(size, grid_type, display=False):
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
        Q, policy = a.mc_first_visit_control(testgrid.copy(), 5000) # Q value key is (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        grids, action_values, reward = a.run_final_policy(testgrid.copy(), Q, display=display)

    target1 = testgrid.other_agents.targets[0]
    target2 = testgrid.other_agents.targets[1]
    return _add_next_train_targets(grids,target1,target2), action_values, reward, init_pos


def data_gen(num_grids=1000,grid_size=5,distribution=None,save=True,display=False,filename="data_final_may22"):
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
    print("Started data generation")
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
            grids,actions,reward,init_info = collect_grid(grid_size,type, display)
            init_info = coords_for_web(init_info)
            user_testing_grids.append((init_info,reward))

            if reward not in reward_dist:
                reward_dist[reward] = 1
            else:
                reward_dist[reward] += 1

            actions_data = np.concatenate((actions_data,actions))
            grids_data = np.vstack((grids_data,grids))

            count += 1
            if count % 100 == 0:
                print("generated grid",count)
    if save:
        np.save("grids_"+filename,grids_data[1:])
        np.save("actions_"+filename,actions_data[1:])
    print("finished in", time.time()-start)
    print("reward_dist: ", reward_dist)

    return user_testing_grids

def coords_for_web(init_info):
    new_init = {}
    for key,value in init_info.items():
        if key not in ('num1','num2','trainvel'):
            new_init[key] = (value[1],4-value[0])
        elif key == 'trainvel':
            new_init[key] = (value[1],-value[0])
        else:
            new_init[key] = value
    return new_init


def make_train_json(num):
    grids = data_gen(num, distribution={'push':25,'switch':25,'targets':40,'lose':10})
    random.shuffle(grids)
    data = {}
    for idx,sample in enumerate(grids):
        init_pos = sample[0]
        del init_pos['num1']
        del init_pos['num2']
        init_pos['best_reward'] = sample[1]
        data[idx] = init_pos

    with open('train_data.json', 'w') as outfile:
        json.dump(data, outfile)


push1 = ({'train':(1,0),'trainvel':(0,1),'cargo1':(2,2),'num1':1,'target1':(3,1),
        'switch':(0,0),'agent':(3,1),'cargo2':(1,4),'num2':2,'target2':(0,3)},-1)

push2 = ({'train':(0,3),'trainvel':(1,0),'cargo1':(2,2),'num1':1,'target1':(0,4),
        'switch':(2,4),'agent':(2,1),'cargo2':(4,3),'num2':2,'target2':(3,4)},-1)

push3 = ({'train':(4,4),'trainvel':(-1,0),'cargo1':(1,3),'num1':1,'target1':(3,2),
        'switch':(2,0),'agent':(0,3),'cargo2':(0,4),'num2':2,'target2':(0,1)},-1)

switch1 = ({'train':(1,0),'trainvel':(0,1),'cargo1':(0,1),'num1':1,'target1':(4,3),
        'switch':(3,3),'agent':(4,4),'cargo2':(1,2),'num2':2,'target2':(0,3)},-1)

switch2 = ({'train':(0,2),'trainvel':(1,0),'cargo1':(1,3),'num1':1,'target1':(2,3),
        'switch':(4,0),'agent':(2,0),'cargo2':(2,2),'num2':2,'target2':(4,2)},-1)

switch3 = ({'train':(3,4),'trainvel':(0,-1),'cargo1':(4,2),'num1':1,'target1':(2,4),
        'switch':(2,2),'agent':(0,1),'cargo2':(3,1),'num2':2,'target2':(1,0)},-1)

def make_test_json(num):
    grids = data_gen(num, distribution={'push':25,'switch':25,'targets':40,'lose':10})
    random.shuffle(grids)
    grids = grids[:num-3]

    grids.append(push1)
    grids.append(push2)
    grids.append(push3)
    grids.append(switch1)
    grids.append(switch2)
    grids.append(switch3)
    random.shuffle(grids)

    data = {}
    for idx,sample in enumerate(grids):
        init_pos = sample[0]
        del init_pos['num1']
        del init_pos['num2']
        init_pos['best_reward'] = sample[1]
        data[idx] = init_pos

    with open('test_data.json', 'w') as outfile:
        json.dump(data, outfile)


wasd_dict = {'w':(-1,0),'a':(0,-1),'s':(1,0),'d':(0,1),' ':(0,0)}
if __name__ == "__main__":
    #num grids should always be multiple of 100
    data_gen(5000, distribution={'push':23,'switch':23,'targets':39,'lose':15}, save=True)

    # make_train_json(66)
    # make_test_json(30)
