import numpy as np

class cliff_walking():
    def __init__(self):
        dimW, dimH = 12, 4
        self._dims = (dimW,dimH)
        self.num_states, self.num_actions = dimW*dimH, 4
        self.state_dim, self.action_dim = 2, 1
        self._states = []
        for i in range(dimH):
            for j in range(dimW):
                self._states.append([i,j])

        self._target = dimW*dimH -1
        self._cliff = list(np.arange(37,47))
        reserved_pos = self._cliff+[self._target]
        self._possible_positions = np.delete(np.arange(0,dimW*dimH),reserved_pos)
        self.reset()

    def reset(self,random_initial=False):
        if not random_initial:
            self._agent = 36
        else:
            self._agent = np.random.choice(self._possible_positions)
        return self._agent

    def observe(self):
        return self._agent, self._target

    def get_coords(self):
        coords = self._states[self._agent]
        return np.array(coords).reshape(self.state_dim,1)

    def step(self,action):
        assert 0<=action<=3, "Invalid action. The action must be an integer within [0,3]"
        self.move(action)
        state,target = self.observe()
        reward = self.reward()
        done = True if (state==target or state in self._cliff) else False
        return state, reward, done

    def move(self,action):
        #ACTIONS: 0-UP, 1-RIGHT, 2-DOWN, 3-LEFT
        y,x = self._states[self._agent]
        if action==0:
            new_x = x
            new_y = y-1
        elif action==1:
            new_x = x+1
            new_y = y
        elif action==2:
            new_x = x
            new_y = y+1
        elif action==3:
            new_x = x-1
            new_y = y
        new_x = np.clip(new_x,0,self._dims[0]-1)
        new_y = np.clip(new_y,0,self._dims[1]-1)
        self._agent = self._states.index([new_y,new_x])

    def reward(self):
        state,target = self.observe()
        if(state==target):
            reward = 10.0
        elif(state in self._cliff):
            reward = -1.0
        else:
            reward = -0.1
        return reward

    def get_world_display(self):
        self._world = np.zeros((self._dims[1],self._dims[0]))
        for cell in self._cliff:
            y,x = self._states[cell]
            self._world[y,x] = 1
        y,x = self._states[self._agent]
        self._world[y,x] = 3
        y,x = self._states[self._target]
        self._world[y,x] = 2
        return self._world


class grid_world():
    def __init__(self):
        dim = 8
        self._dims = (dim,dim)
        self.num_states = dim*dim
        self.num_actions = 4
        self._target = [self._dims[1]-1,self._dims[0]-1]
        self._obstacles = [11,12,19,20,25,26,27,28,29,30,33,34,35,36,37,38,43,44,51,52]
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

    def get_coords(self):
        return np.array(self._agent).reshape(2,1)

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

    def get_world_display(self,print_map=False):
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
