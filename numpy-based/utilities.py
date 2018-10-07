import numpy as np
import matplotlib.pyplot as plt


def plot_policies(env,policies,size=(10,4)):
    cols = 5
    rows = int(np.ceil((len(policies)+1)/cols))
    print('______________________')
    print('UP, RIGHT, DOWN, LEFT')
    plt.figure(figsize=(2,1))
    plt.imshow(np.reshape(list(np.arange(0,4)),(1,4)));
    plt.axis('off');
    plt.figure(figsize = size)
    for i,policy in enumerate(policies):
        plt.subplot(rows,cols,i+1)
        plt.imshow(policy.reshape(env._dims[1],env._dims[0]));
        plt.axis('off');
        