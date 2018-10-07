import numpy as np

class cliff_walking():
    def __init__(self):
        dimW = 12
        dimH = 4
        self._dims = (dimW,dimH)
        self.num_states = dimW*dimH
        self.num_actions = 4
        self._target = [self._dims[0]-1,self._dims[1]-1]
        self._cliff = list(np.arange(37,47))
        target_pos = self.encode_state(self._target)
        reserved_pos = self._cliff+[target_pos]
        self._possible_positions = np.delete(np.arange(0,self._dims[0]*self._dims[1]),reserved_pos)
        self.reset()

    def reset(self,initial_pos='standar'):
        self._agent = [0,self._dims[1]-1]
        if initial_pos=='random_2':
            if np.random.random() >= 0.5:
                self._agent = [0,self._dims[1]-1]
            else:
                self._agent = [0,0]
        elif initial_pos=='random_all':
            x,y = self.decode_coords(np.random.choice(self._possible_positions))
            self._agent = [x,y]
        return self.encode_state(self._agent)

    def observe(self):
        return self.encode_state(self._agent), self.encode_state(self._target)
    
    def observe_coordinates(self):
        return self._agent

    def step(self,action):
        assert 0<=action<=3, "Invalid action. The action must be an integer within [0,3]"
        self.move(action)
        state,target = self.observe()       
        reward = self.reward()
        done = True if (state==target or state in self._cliff) else False
        return state, reward, done

    def move(self,action):
        #ACTIONS: 0-UP, 1-RIGHT, 2-DOWN, 3-LEFT
        if action==0:
            new_x = self._agent[0]
            new_y = self._agent[1]-1
        elif action==1:
            new_x = self._agent[0]+1
            new_y = self._agent[1]
        elif action==2:
            new_x = self._agent[0]
            new_y = self._agent[1]+1
        elif action==3:
            new_x = self._agent[0]-1
            new_y = self._agent[1]
        self._agent[0] = np.clip(new_x,0,self._dims[0]-1)
        self._agent[1] = np.clip(new_y,0,self._dims[1]-1)
            
    def reward(self):
        state,target = self.observe()
        if(state==target):
            reward = 1.0
        elif(state in self._cliff):
            reward = -10.0
        else:
            reward = -1.0
        return reward

    def encode_state(self,coords):
        return coords[0]+(coords[1])*self._dims[0]

    def decode_coords(self,state):
        ver = state//(self._dims[0])
        hor = state - self._dims[1]*ver*(self._dims[0]/self._dims[1])
        return int(hor),int(ver)

    def render(self,print_map=False):
        self._world = np.zeros((self._dims[1],self._dims[0]))
        for cell in self._cliff:
            x,y = self.decode_coords(cell)
            self._world[y,x]=1
        self._world[self._agent[1],self._agent[0]] = 3
        self._world[self._target[1],self._target[0]] = 2
        if print_map:
            print('__________________________')
            print(self._world)
        return self._world


class grid_world():
    def __init__(self):
        dim = 8
        self._dims = (dim,dim)
        self.num_states = dim*dim
        self.num_actions = 4
        self._target = [self._dims[1]-1,self._dims[0]-1]
        self._obstacles = [11,12,19,20,25,26,27,28,29,30,
                           33,34,35,36,37,38,43,44,51,52]
        target_pos = self.encode_state(self._target)
        reserved_pos = self._obstacles+[target_pos]
        self._possible_positions = np.delete(np.arange(0,self._dims[0]*self._dims[1]),reserved_pos)
        self.reset()

    def reset(self,random_initial=False):
        self._agent = [0,0]
        if random_initial:
            y,x = self.decode_coords(np.random.choice(self._possible_positions))
            self._agent = [x,y]
        return self.encode_state(self._agent)

    def observe(self):
        return self.encode_state(self._agent), self.encode_state(self._target)
    
    def observe_coordinates(self):
        return self._agent

    def step(self,action):
        assert 0<=action<=3, "Invalid action. The action must be an integer within [0,3]"
        self.move(action)
        state,target = self.observe()       
        reward = self.reward()
        done = True if state==target else False
        return state, reward, done

    def move(self,action):
        #ACTIONS: 0-UP, 1-RIGHT, 2-DOWN, 3-LEFT
        if action==0:
            new_x = self._agent[0]
            new_y = self._agent[1]-1
        elif action==1:
            new_x = self._agent[0]+1
            new_y = self._agent[1]
        elif action==2:
            new_x = self._agent[0]
            new_y = self._agent[1]+1
        elif action==3:
            new_x = self._agent[0]-1
            new_y = self._agent[1]
        new_state = [new_x,new_y]
        if (self.encode_state(new_state) not in self._obstacles):
            self._agent[0] = np.clip(new_x,0,self._dims[0]-1)
            self._agent[1] = np.clip(new_y,0,self._dims[1]-1)
            
    def reward(self):
        return 10.0 if (self.encode_state(self._agent)==self.encode_state(self._target)) else -1.0

    def encode_state(self,coords):
        return coords[0]+(coords[1])*self._dims[1]

    def decode_coords(self,state):
        ver = state//(self._dims[0])
        hor = state - self._dims[1]*ver
        return ver,hor

    def render(self,print_map=False):
        self._world = np.zeros(self._dims)
        for obstacle in self._obstacles:
            y,x = self.decode_coords(obstacle)
            self._world[y,x]=1
        self._world[self._agent[1],self._agent[0]] = 3
        self._world[self._target[1],self._target[0]] = 2
        if print_map:
            print('__________________________')
            print(self._world)
        return self._world
    

