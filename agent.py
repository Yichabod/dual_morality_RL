import numpy as np
from collections import Counter, defaultdict
import neural_net
from graphics import display_grid
from utils import generate_array, in_bounds
import new_grid as grid
#import torch

ACTION_DICT = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4} #

class Agent:
    """
    Dual processing agent.
    Given a set of actions and a view of the grid, decides what action to take
    """


    def __init__(self):
        pass

    def train_load_neural_net(self):
        """
        Attempt to load neural net from 'neural_net' file. If not present, train neural net
        and then return the network
        """
        try:
            return self.net
        except(AttributeError):
            try:
                net = neural_net.load()
                print("neural net loaded")
            except:
                print("training neural net")
                neural_net.train()
                net = neural_net.load()
            self.net = net
            return net

    def neural_net_output(self, grid):
        net = self.train_load_neural_net()

        state_array = generate_array(grid)[0,:,:] #(1,5,5) -> (5,5)
        next_train = np.zeros((grid.size,grid.size),dtype=int)
        next_train_y = grid.train.pos[0]+grid.train.velocity[0]
        next_train_x = grid.train.pos[1]+grid.train.velocity[1]
        if in_bounds(grid.size, (next_train_y,next_train_x)):
            next_train[next_train_y][next_train_x] = 1
        test_input = np.stack((state_array,next_train))

        out = neural_net.predict(net, test_input)
        return out[0]

    def run_model_free_policy(self, grid, display=False):
        """
        Use neural network to solve MDP (no exploration)
        args:
        grid = grid object
        display = whether to display the grid
        return_grid_type = whether t
        returns:
        np array of grids where agent has taken action according to policy
        np array of actions that agent took
        type of grid (agent hit by train, agent push switch, agent push others, train hit others): str
        """
        if display: display_grid(grid)
        net = self.train_load_neural_net()

        total_reward = 0

        while not grid.terminal_state: # max number of steps per episode
            # grids.append(state)
            state_array = generate_array(grid)[0,:,:] #(1,5,5) -> (5,5)

            #updates next train pos input grid for neural_net.predict function
            next_train = np.zeros((grid.size,grid.size),dtype=int)
            next_train_y = grid.train.pos[0]+grid.train.velocity[0]
            next_train_x = grid.train.pos[1]+grid.train.velocity[1]
            if in_bounds(grid.size, (next_train_y,next_train_x)):
                next_train[next_train_y][next_train_x] = 1
            test_input = np.stack((state_array,next_train))

            action_ind = np.argmax(neural_net.predict(net, test_input))
            action = grid.all_actions[action_ind]
            if display: print(neural_net.predict(net, test_input))
            if display: print(action)

            total_reward += grid.R(action)
            grid.T(action)
            if display: display_grid(grid)
        return total_reward

    def _create_softmax_policy(self,Q_dict,cutoff=0, nn_init = False):
        def policy(grid):
            state = grid.current_state

            if state not in Q_dict and nn_init:
                Q_dict[state] = self.neural_net_output(grid)

            Q_values = Q_dict[state]
            e_x = np.exp(Q_values - np.max(Q_values))
            softmax = e_x / e_x.sum()
            for ind,action_prob in enumerate(softmax):
                if action_prob<cutoff:
                    softmax[ind] = 0
            if np.count_nonzero(softmax) == 0:
                return [1/(len(Q_values)) for k in range(len(Q_values))]
            return softmax/np.sum(softmax)
        return policy

    def _create_epsilon_greedy_policy(self, Q_dict, epsilon=0.2, nn_init = False):
        """
        Use Q_dict to create a greedy policy
        args: Q_dict[state] = [value of action1, value of action2, ...]
                where state is defined as result of grid.current_state (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
        returns: policy function takes in state and chooses with prob 1-e+(e/|A|) maxQ value action
        """
        def policy(grid):
            state = grid.current_state

            if state not in Q_dict and nn_init:
                Q_dict[state] = self.neural_net_output(grid)

            Q_values = Q_dict[state]

            action_probs = [0 for k in range(len(Q_values))]
            best_action = np.argmax(Q_values)
            for i in range(len(Q_values)):
                if np.count_nonzero(Q_values) == 0: #all are zero
                    action_probs[i] = 1/len(Q_values)
                elif i == best_action:
                    action_probs[i] = 1-epsilon+(epsilon/len(Q_values))
                else:
                    action_probs[i] = epsilon/len(Q_values)
            return action_probs
        return policy

    def run_final_policy(self, grid, Q_dict, nn_init=False, display=False):
        """
        Use Q_dict to solve MDP (no exploration)
        args: grid = original grid state, Q_dict[state] = [value of action1, value of action2, ...]
        nn_init = whether we want to use the neural network to initialise the monte carlo policy
        returns: 2 numpy arrays containing grid and optimal action pairs to be
        fed into downstream model
        """
        policy = self._create_epsilon_greedy_policy(Q_dict,epsilon=0,nn_init=nn_init) #optimal policy, eps=0 always chooses best value
        #if display: display_grid(grid)
        state = grid.current_state
        total_reward = 0 #to keep track of most significant action taken by agent
        grids_array = np.empty((1,grid.size,grid.size),dtype=int)
        action_val_array = np.empty((1,grid.size),dtype=int)

        while not grid.terminal_state: # max number of steps per episode
            action_probs = policy(grid)
            action_ind = np.argmax(action_probs)
            if display:
                pass#print(Q_dict[state])
            action = grid.all_actions[action_ind]
            if display: display_grid(grid)
            if display: print(action)
            action_val_array = np.concatenate((action_val_array,np.array([Q_dict[grid.current_state]])))
            grids_array = np.vstack((grids_array,generate_array(grid)))
            total_reward += grid.R(action)
            newstate = grid.T(action)
            state = newstate

        if display: print(total_reward)
        return grids_array[1:], action_val_array[1:], total_reward


    def mc_first_visit_control(self, start_grid, iters, discount_factor=0.9, epsilon=0.2, nn_init=False, cutoff = 0, softmax = True) -> tuple:
        """
        Monte Carlo first visit control. Uses epsilon greedy strategy
        to find optimal policy. Details can be found page 101 of Sutton
        Barto RL Book
        Args: nn_init whether to initialize Q-values with neural net outputs
        Returns: (Q_values, policy)
                Q(s,a) = val, policy(state) = action
        """
        # Q is a dictionary mapping state to [value of action1, value of action2,...]
        grid = start_grid.copy()

        if nn_init:
            Q = {}
        else:
            Q = defaultdict(lambda: list(0 for i in range(len(grid.all_actions))))

        if softmax:
            policy = self._create_softmax_policy(Q, cutoff, nn_init)
        else:
            policy = self._create_epsilon_greedy_policy(Q,epsilon, nn_init)

        sa_reward_sum, total_sa_counts = defaultdict(int), defaultdict(int) #keep track of total reward and count over all episodes
        for n in range(iters):
            # generate episode
            episode = []
            grid = start_grid.copy() #copy because running episode mutates grid object
            state = grid.current_state
            while not grid.terminal_state: # max number of steps per episode
                action_probs = policy(grid)

                action_ind = np.random.choice(np.arange(len(action_probs)), p=action_probs)
                action = grid.all_actions[action_ind]

                #must calculate reward before transitioning state, otherwise reward will be calculated for action in newstate
                reward = grid.R(action)
                newstate = grid.T(action)
                episode.append((state, action, reward))
                state = newstate
                #display_grid(grid)
            sa_counts = Counter([(x[0],x[1]) for x in episode]) #dictionary: [s,a]=count
            G = 0 #averaged reward
            for t in range(len(episode)-1,-1,-1):
                G = discount_factor*G + episode[t][2] #reward at the next time step
                state = episode[t][0]
                action = episode[t][1]
                action_index = grid.all_actions.index(action)
                sa_pair = state, action
                sa_counts[sa_pair] -= 1
                if sa_counts[sa_pair] == 0: #appears for the first time
                    sa_reward_sum[sa_pair] += G
                    total_sa_counts[sa_pair] += 1
                    Q[state][action_index] = sa_reward_sum[sa_pair]/total_sa_counts[sa_pair] #average reward over all episodes
                    if softmax:
                        policy = self._create_softmax_policy(Q,cutoff,nn_init)
                    else:
                        policy = self._create_epsilon_greedy_policy(Q, epsilon,nn_init)

        return Q, policy


if __name__ == "__main__":

    #push_init_pos = {'train':(2,0),'agent':(4,1),'other1':(3,2),'switch':(0,0),'other2':(2,4),'other1num':1,'other2num':4}
    #switch_init_pos = {'train':(2,0),'agent':(4,1),'other1':(0,0),'switch':(3,2),'other2':(2,4),'other1num':1,'other2num':4}

    #here agent only needs to push cargo into target, which is near train. To test model free
    easy1 = {'train':(1,0),'trainvel':(0,1),'cargo1':(3,2),'num1':1,'target1':(2,2),
            'switch':(0,0),'agent':(4,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}
    # agent should push simple cargo into target, away from train
    easy2 = {'train':(1,0),'trainvel':(0,1),'cargo1':(4,1),'num1':1,'target1':(4,0),
            'switch':(0,0),'agent':(2,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}
    # agent should get out of the way of the train
    easy3 = {'train':(1,0),'trainvel':(0,1),'cargo1':(4,1),'num1':1,'target1':(4,0),
            'switch':(0,0),'agent':(1,2),'cargo2':(2,4),'num2':2,'target2':(0,3)}

    #somewhere between 10,000 and 50,000 iterations the mc finally gets it - seems pretty hard without nn even
    push3 = {'train':(1,0),'trainvel':(0,1),'cargo1':(2,3),'num1':1,'target1':(3,1),
            'switch':(4,0),'agent':(3,3),'cargo2':(2,4),'num2':2,'target2':(1,4)}

    #this one takes a long time too - perhaps because reward comes too late
    death1 = {'train':(0,0),'trainvel':(0,1),'cargo1':(1,2),'num1':1,'target1':(2,2),
            'switch':(4,0),'agent':(0,3),'cargo2':(2,4),'num2':2,'target2':(3,3)}

    push1 = {'train':(1,0),'trainvel':(0,1),'cargo1':(2,2),'num1':1,'target1':(3,1),
        'switch':(0,0),'agent':(3,3),'cargo2':(1,4),'num2':2,'target2':(0,3)}

    targets_test = {'train':(0,0),'trainvel':(0,1),'other1':(1,2),'num1':1,'target1':(1,3),
            'switch':(4,4),'agent':(2,1),'other2':(2,2),'num2':2,'target2':(3,2)}

    weird1 = {'train':(4,2),'trainvel':(-1,0),'other1':(4,3),'num1':1,'target1':(0,3),
            'switch':(1,0),'agent':(3,0),'other2':(2,2),'num2':2,'target2':(2,4)}

    switch = {'train':(1,0),'trainvel':(0,1),'cargo1':(2,1),'num1':1,'target1':(4,3),
            'switch':(3,3),'agent':(4,4),'cargo2':(1,2),'num2':2,'target2':(0,3)}


    push4 = {'train':(2,0),'trainvel':(0,1),'cargo1':(3,2),'num1':1,'target1':(0,1),
        'switch':(4,0),'agent':(4,2),'cargo2':(3,3),'num2':2,'target2':(0,3)}

    testgrid = grid.Grid(5,random=False, init_pos=easy3)

    agent = Agent()

    model = 'free'
    if model == 'dual':
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 20, nn_init=True,cutoff=0.4,softmax = True)
        agent.run_final_policy(testgrid.copy(), Q,nn_init=True,display=True)
    if model == 'free':
        agent.run_model_free_policy(testgrid.copy(),display=True)
    if model == 'based':
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), iters=10000, nn_init=False, softmax=False)
        #display_grid(testgrid.copy())
        agent.run_final_policy(testgrid.copy(), Q,nn_init=False,display=True)
