import numpy as np
import matplotlib.pyplot as plt

def plot_policies(policies,idx=(50,30),size=(15,15),dims=(8,8)):
    assert len(policies)>=idx[0], "The number of policies is less than the number of episodes to sample from (default: 50)."
    assert idx[0]>=idx[1], "The number of episodes to sample from is smaller than the number of policies to plot."
    cols = 5
    rows = int(np.ceil((idx[1]+1)/cols))
    print('______________________')
    print('UP, RIGHT, DOWN, LEFT')
    plt.figure(figsize=(2,1))
    plt.imshow(np.reshape(list(np.arange(0,4)),(1,4)),cmap='viridis',vmin=0,vmax=3);
    plt.axis('off');

    policies2plot = []
    indexes = list(np.linspace(0,idx[0],idx[1]-1).astype(int))+[len(policies)-1]
    for j in indexes:
        policies2plot.append(policies[j])

    plt.figure(figsize = size)
    for i,policy in enumerate(policies2plot):
        plt.subplot(rows,cols,i+1)
        plt.imshow(policy.reshape(dims[1],dims[0]),cmap='viridis',vmin=0,vmax=3);
        plt.axis('off');
        plt.title('Policy episode: '+str(indexes[i]));
        
def plot_trayectory(trayectory,size=(15,10)):
    num = len(trayectory)
    cols = 5
    rows = int(np.ceil((num)/cols))
    plt.figure(figsize=size)
    for i,state in enumerate(trayectory):
        plt.subplot(rows,cols,i+1)
        plt.imshow(state,cmap='viridis',vmin=0,vmax=3);
        plt.axis('off');
        
def print_actions_taken(actions):
    action_names = ['UP','RIGHT','DOWN','LEFT']
    for i,a in enumerate(actions):
        print('Step:',i+1,'   Action taken:',action_names[a])
        
def extract_policy(env,agent):
    policy = np.zeros((env._dims[1],env._dims[0]))
    for pos in env._possible_positions:
        state = np.array(env._states[pos]).reshape(env.state_dim,1)
        action = agent.greedy(state)
        policy[state[0],state[1]] = action
    return policy
        
def plot_policies_DQN(policies,size=(8,4)):
        
    print('______________________')
    print('UP, RIGHT, DOWN, LEFT')
    plt.figure(figsize=(2,1))
    plt.imshow(np.reshape(list(np.arange(0,4)),(1,4)),cmap='viridis',vmin=0,vmax=3);
    plt.axis('off');
    
    plt.figure(figsize=size)
    plt.subplot(1,2,1)
    plt.imshow(policies[0],cmap='viridis',vmin=0,vmax=3);
    plt.axis('off');
    plt.title('Initial policy');
    plt.subplot(1,2,2)
    plt.imshow(policies[1],cmap='viridis',vmin=0,vmax=3);
    plt.axis('off');
    plt.title('Current policy');
    

        