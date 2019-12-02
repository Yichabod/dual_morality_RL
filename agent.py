import numpy as np
from collections import Counter, defaultdict
import neural_net
from graphics import display_grid
from utils import generate_array

ACTION_DICT = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4}

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
        # try:
        #     net = neural_net.load()
        # except:
        neural_net.train()
        net = neural_net.load()
        return net

    def run_model_free_policy(self, grid, display=False):
        """
        Use neural network to solve MDP (no exploration)
        args: grid = original grid state, neural_net = trained neural net, used to choose next action
        returns: 2 numpy arrays containing grid and optimal action pairs to be
        fed into downstream model
        """
        if display: display_grid(grid)
        # state = grid.current_state
        grids_array = np.empty((1,grid.size,grid.size),dtype=int)
        grids = []
        actions = []
        net = self.train_load_neural_net()
        while not grid.terminal_state: # max number of steps per episode
            # grids.append(state)
            state_array = generate_array(grid)[0,:,:] #(1,5,5) -> (5,5)
            action_ind = np.argmax(neural_net.predict(net, state_array))
            action = grid.all_actions[action_ind]
            if display: print(action)
            actions.append(action)
            grids.append(generate_array(grid))
            grid.T(action)
            if display: display_grid(grid)
        return np.array(grids), np.array(actions)


    def _create_epsilon_greedy_policy(self, Q_dict, nn_initialization=False, epsilon=0.2):
        """
        Use Q_dict to create a greedy policy
        args: Q_dict[state] = [value of action1, value of action2, ...]
                where state is defined as result of grid.current_state (self.agent_pos,self.train.pos,list(self.other_agents.positions)[0])
              nn_initialization = whether to use neural network to seed the Q value dictionary
              TODO
        returns: policy function takes in state and chooses with prob 1-e+(e/|A|) maxQ value action
        """
        def policy(state):
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

    def run_final_policy(self, grid, Q_dict, display=False):
        """
        Use Q_dict to solve MDP (no exploration)
        args: grid = original grid state, Q_dict[state] = [value of action1, value of action2, ...]
        returns: 2 numpy arrays containing grid and optimal action pairs to be
        fed into downstream model
        """
        policy = self._create_epsilon_greedy_policy(Q_dict,epsilon=0) #optimal policy, eps=0 always chooses best value
        if display: display_grid(grid)
        state = grid.current_state
        grids_array = np.empty((1,grid.size,grid.size),dtype=int)
        action_val_array = np.empty((1,grid.size),dtype=int)
        while not grid.terminal_state: # max number of steps per episode
            action_probs = policy(state)
            action_ind = np.argmax(action_probs)
            action = grid.all_actions[action_ind]
            if display: print(action)

            ACTION_DICT = {(0, 0):0, (-1, 0):1, (0, 1):2, (1, 0):3, (0, -1):4}
            action_val_array = np.concatenate((action_val_array,np.array([Q_dict[grid.current_state]])))
            grids_array = np.vstack((grids_array,generate_array(grid)))

            newstate = grid.T(action)
            state = newstate
            if display: display_grid(grid)
        return grids_array[1:], action_val_array[1:]


    def mc_first_visit_control(self, start_grid, n_episodes, discount_factor=0.9, epsilon=0.2) -> tuple:
        """
        Monte Carlo first visit control. Uses epsilon greedy strategy
        to find optimal policy. Details can be found page 101 of Sutton
        Barto RL Book
        Args: mdp - Grid class
        Returns: (Q_values, policy)
                Q(s,a) = val, policy(state) = action
        """
        # Q is a dictionary mapping state to [value of action1, value of action2,...]
        grid = start_grid.copy()
        Q = defaultdict(lambda: list(0 for i in range(len(grid.all_actions))))
        policy = self._create_epsilon_greedy_policy(Q, epsilon) #initialized random policy
        sa_reward_sum, total_sa_counts = defaultdict(int), defaultdict(int) #keep track of total reward and count over all episodes
        for n in range(n_episodes):
            # generate episode
            episode = []
            grid = start_grid.copy() #copy because running episode mutates grid object
            state = grid.current_state
            while not grid.terminal_state: # max number of steps per episode
                action_probs = policy(state)
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
                    policy = self._create_epsilon_greedy_policy(Q, epsilon)
        return Q, policy

if __name__ == "__main__":
    import grid
    testgrid = grid.Grid(5,random=False)
    agent = Agent()
    model_based = True
    if model_based == True:
        Q, policy = agent.mc_first_visit_control(testgrid.copy(), 1000)
        print(agent.run_final_policy(testgrid.copy(), Q,display=True))
    else:
        print(agent.run_model_free_policy(testgrid.copy(),display=True))
