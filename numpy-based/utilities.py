import numpy as np
import matplotlib.pyplot as plt

def plot_policies(policies,idx=(50,30),size=(15,10),dims=(8,8)):
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
        