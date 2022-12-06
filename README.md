# KoopmanNetwork
Train AI to read your brain! 🤖➡️🧠

This repo is active! Work for this project is based on research from [Dr. Bethany Lusch](https://github.com/BethanyL). This project is supervised by [Max Kanwal](https://neuroscience.stanford.edu/people/max-kanwal) at Prof. Kwabena Boahen's [Brains in Silicon](http://web.stanford.edu/group/brainsinsilicon/) lab at Stanford.



![Alt Text](https://github.com/FlyingWorkshop/KoopmanNetwork/blob/main/auto_1.gif)

## Installation
First, install the required packages with
```
pip install -r requirements.txt
```

## Quickstart

### Data Generation

First create an `LDS` (linear dynamical system) instance. We use this object to create our training and testing data.
Training and testing data is an array of trajectories (a trajectory is an array of coordinates). To generate trajectories,
we call the `.make_rand_trajs()` method off of our `LDS` instance. This is an extremely efficient method of creating `5000` trajectories
for our given `A` matrix with random initial conditions between `-50` and `0`. Notice that when we generate our test data, we pick initial conditions outside the region that we draw our training data from. We also add noise to our training data so that it's harder to learn (and more similar to nonlinear data). Finally, we plot `10` trajectories from `true_train, noisy_train, test`. The `plot` function is extremely robust. It can plot trajectories in $\mathbb{R}^n$ for $n \geq 2$. For trajectories, in more than three dimensions, we use PCA to view the lower dimensional projection of the high dimensional trajectories.
```
lds = kp.dynamics.LDS(A=[[-2, 0, 4], [4, 0, 4], [-4, -2, -4]])
true_train, noisy_train = lds.make_rand_trajs(0, 50, 5000, noise_var=1)
test = lds.make_rand_trajs(-50, 0, 1000, noise_var=0)
kp.utils.plot([true_train, noisy_train, test],
              target_dim=3,
              labels=["true train", "noisy train", "test"],
              max_lines=10)
```

### Training

Training is extremely simple. 
```
net = kp.network.KoopmanNetwork(lds.dim)
net.train(noisy_train, autoencoder_epochs=50, autoencoder_batch_size=100, model_epochs=30, model_batch_size=10)
```

