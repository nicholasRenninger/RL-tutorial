from agents.DDPG.DDPG_agent import actor_network, critic_network, DDPG_agent
from agents.DDPG.RL_utils import ReplayBuffer, OrnsteinUhlenbeckActionNoise
import gym
import numpy as np

env = gym.make('Pendulum-v0')
actor = actor_network([3,400,300,1])
critic = critic_network([3,400,300,1])
buffer = ReplayBuffer()
noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(1))
agent = DDPG_agent(actor,critic)

max_episodes = 100
max_steps = 1000

grads = 0
for episode in range(max_episodes):
    accum_rw = 0
    state = env.reset()
    state = state.reshape(3,1)
    for step in range(max_steps):
        if(episode>=max_episodes-10):
            env.render()

        action = 2.0*agent.actor(state)[-1] + 1./(1.+episode+step)
        next_state, reward, done,_ = env.step(action)
        #print(reward)
        next_state = next_state.reshape(3,1)

        buffer.add(state, action, next_state, reward, done)

        state = next_state
        accum_rw += reward
        if(buffer.size()>1000):
            batch= buffer.sample_batch(32)
            grads = agent.train(batch)
            agent.update_targets()

    if(episode%(max_episodes/10)==0):
        print('Episode: ',episode,'  Accumulated reward: ',accum_rw)


env.close()
