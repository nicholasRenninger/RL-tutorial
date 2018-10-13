from envs.environments import cliff_walking, grid_world
from agents.Qlearning import double_deep_Qlearning, ReplayBuffer
import numpy as np
import matplotlib.pyplot as plt


env = cliff_walking()
ns = env.num_states
na = env.num_actions
dims = [2,10,10,4]
np.random.seed(123)
agent = double_deep_Qlearning(dims,learning_rate=1e-2)
buffer = ReplayBuffer(100000,2,1)


Ntrain = 2000
Nsteps = 100
plt.ion()
window = plt.figure()
for episode in range(Ntrain+1):
    env.reset()
    done = False
    step = 0
    loss = np.zeros((4,1))
    agent.epsilon_decay(0.999)
    while(not done and step<=Nsteps):
        state = env.get_coords()
        action = agent.epsilon_greedy(state)
        _,reward,done = env.step(action)
        next_state = env.get_coords()
        buffer.add(state,action,next_state,reward,done)
        if(buffer.size()>1000):
            batch = buffer.sample_batch(32)
            loss = agent.train(batch)
            agent.update_target_weights()

        step += 1
        if(episode>=Ntrain-5):
            map_env = env.get_world_display()
            plt.clf()
            plt.imshow(map_env)
            window.canvas.draw()
            window.canvas.flush_events()
    if(episode%(Ntrain/10)==0):
        print(episode,step,loss.mean(),agent._epsilon)


plt.ion()
window = plt.figure()
for i in range(10):
    env.reset()
    done = False
    step=0
    while (not done and step <= 30):
        map_env = env.get_world_display()
        state = env.get_coords()
        action = agent.greedy(state)
        _,_,done=env.step(action)
        step+=1
        plt.clf()
        plt.imshow(map_env)
        window.canvas.draw()
        window.canvas.flush_events()
