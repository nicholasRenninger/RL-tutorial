import numpy as np

class tabular_Qlearning():
    def __init__(self,num_states,num_actions,lr=0.1):
        self._num_states = num_states
        self._num_actions = num_actions
        self._gamma = 0.99
        self._initial_alpha = lr
        self.reset()

    def reset(self,initial_epsilon=1.0):
        self.Qtable = np.zeros((self._num_states,self._num_actions))
        self._epsilon = initial_epsilon
        self._alpha = self._initial_alpha
        
    def greedy(self,state):
        return np.argmax(self.Qtable[state,:])

    def epsilon_greedy(self,state):
        if np.random.random() <= self._epsilon:
            return np.random.randint(0,4)
        else:
            return self.greedy(state)
        
    def epsilon_decay(self,rate=0.99,min=0.1):
        self._epsilon = np.clip(rate*self._epsilon,min,1.0)
    
    def alpha_decay(self,rate=0.99,min=1e-3):
        self._alpha = np.clip(rate*self._alpha,min,self._initial_alpha)

    def train(self,state,action,next_state,reward,done):
        if not done:
            BE = reward + self._gamma*np.amax(self.Qtable[next_state,:]) - self.Qtable[state,action]
        else:
            BE = reward - self.Qtable[state,action]
        self.Qtable[state,action] += self._alpha*BE
        return 0.5*np.square(BE)
    
    def get_policy(self):
        return np.argmax(self.Qtable,axis=1)

class nn_Qlearning():
    def __init__(self,neural_network):
        self.nn = neural_network
        self._gamma = 0.99
        self.reset()

    def reset(self):
        self.nn.reset()
        self._epsilon = 1.0
        self._alpha = 0.5
        
    def greedy(self,state):
        prediction = self.nn.forward_pass(state)
        return np.argmax(prediction)

    def epsilon_greedy(self,state):
        if np.random.random() <= self._epsilon:
            return np.random.randint(0,4)
        else:
            return self.greedy(state)
        
    def epsilon_decay(self,rate=0.99,min=0.1):
        self._epsilon = np.clip(rate*self._epsilon,min,1.0)

    def train(self,state,action,next_state,reward,done):
        if not done:
            target = reward + self._gamma*np.amax(self.nn.forward_pass(next_state))
        else:
            target = reward
        target_f = self.nn.forward_pass(state)
        target_f[action] = target
        BE = self.nn.backward_pass(target_f)
        return BE
    
    def get_policy(self):
        pass
