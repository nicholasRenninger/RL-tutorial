import numpy as np

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

    def reset(self,random_intial=False):
        self._agent = [0,0]
        if random_intial:
            y,x = self.decode_coords(np.random.choice(self._possible_positions))
            self._agent = [x,y]
        return self.encode_state(self._agent)

    def observe(self):
        return self.encode_state(self._agent), self.encode_state(self._target)

    def step(self,action):
        assert 0<=action<=3, "Invalid action. The action must be an integer within [0,3]"
        self.move(action)
        state = self.encode_state(self._agent)
        reward = 1.0 if (self._agent[0]==self._target[0] and self._agent[1]==self._target[1]) else 0.0
        done = True if reward==1.0 else False
        return state, reward, done, None

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

    def encode_state(self,coords):
        return coords[0]+(coords[1])*self._dims[1]

    def decode_coords(self,state):
        ver = state//(self._dims[0])
        hor = state - self._dims[1]*ver
        return ver,hor

    def render(self):
        self._world = np.zeros(self._dims)
        for obstacle in self._obstacles:
            y,x = self.decode_coords(obstacle)
            self._world[y,x]=3
        self._world[self._agent[1],self._agent[0]] = 1
        self._world[self._target[1],self._target[0]] = 2
        print('__________________________')
        print(self._world)
        return self._world
