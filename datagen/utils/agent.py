import numpy as np
from collections import Counter, defaultdict

from graphics import display_grid

class Agent:
    """
    Dual processing agent.
    Given a set of actions and a view of the grid, decides what action to take
    """
    def __init__(self):
        self.model_based_planner = None

    def _create_epsilon_greedy_policy(self, Q_dict, epsilon=0.2):
        """
        Use Q_dict to create a greedy policy
        args: Q_dict[state] = [value of action1, value of action2, ...]
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
    
    def run_final_policy(self, mdp, policy):
        """
        Use Q_dict to solve MDP (no exploration)
        args: Q_dict[state] = [value of action1, value of action2, ...]
        returns: policy function takes in state and chooses maxQ value action
        """
        grid = mdp(5)
        display_grid(grid)
        state = grid.current_state
        while not grid.terminal_state: # max number of steps per episode
            action_probs = policy(state)
            action_ind = np.argmax(action_probs)
            action = grid.all_actions[action_ind]
            #must calculate reward before transitioning state, otherwise reward will be calculated for action in newstate
            reward = grid.R(action)
            newstate = grid.T(action)
            state = newstate
            display_grid(grid)
        
    
    def mc_first_visit_control(self, mdp, n_episodes, discount_factor=0.9, epsilon=0.2) -> tuple:
        """
        Monte Carlo first visit control. Uses epsilon greedy strategy
        to find optimal policy. Details can be found page 101 of Sutton
        Barto RL Book
        Args: mdp - Grid class
        Returns: (Q_values, policy)
                Q(s,a) = val, policy(state) = action
        """
        # Q is a dictionary mapping state to [value of action1, value of action2,...]
        grid = mdp(5)
        Q = defaultdict(lambda: list(0 for i in range(len(grid.all_actions))))
        policy = self._create_epsilon_greedy_policy(Q, epsilon)
        sa_reward_sum, total_sa_counts = defaultdict(int), defaultdict(int) #keep track of total reward and count over all episodes
        for n in range(n_episodes):
            # generate episode
            episode = []
            grid = mdp(5)
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
    agent = Agent()
    Q, policy = agent.mc_first_visit_control(grid.Grid, 1000)
    agent.run_final_policy(grid.Grid, policy)
