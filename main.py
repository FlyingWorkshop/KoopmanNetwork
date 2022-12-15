import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import tensorflow as tf
from numpy.random import default_rng

import koopman as kp

RNG = default_rng()
CACHE_FOLDER = "cache"


def make_filenames(intrinsic_dim, input_dim, num_trials):
    """
    Helper that returns a list of string filenames. Called before we use multiprocessing, so that we don't
    get weird name errors while saving. Figures out how many '.json' cache files there
    are for a certain (intrinsic_dim, input_dim) combo (makes the subdirectories if it doesn't exist yet)
    and then returns a list of filenames beginning with +1 whatever the last file cached was.

    Example:
        Say our cache folder looks like this:

        cache/1/2
            1.json
            2.json
            ...
            45.json

        Then make_filenames(1, 2, 10) returns

        ["46.json", ..., "55.json"]
    """
    folder = Path(f"{CACHE_FOLDER}/{intrinsic_dim}/{input_dim}")
    folder.mkdir(parents=True, exist_ok=True)
    num_files = len(list(folder.glob('**/*.json')))
    filenames = [f"{CACHE_FOLDER}/{intrinsic_dim}/{input_dim}/{i}.json" for i in
                 range(num_files + 1, num_files + 1 + num_trials)]
    return filenames


def do_trial(intrinsic_dim, input_dim, filename, verbose):
    lds = kp.dynamics.RandomLDS(intrinsic_dim)
    true_train, noisy_train = lds.make_rand_trajs(-50, 50, 5000)
    test = lds.make_rand_trajs(-50, 50, 1000, noise_var=0)
    W = RNG.integers(-5, 5, size=(intrinsic_dim, input_dim))
    true_train = np.array([traj @ W for traj in true_train])
    noisy_train = np.array([traj @ W for traj in noisy_train])
    test = np.array([traj @ W for traj in test])
    net = kp.network.KoopmanNetwork(input_dim, intrinsic_dim)
    # net.train(noisy_train, autoencoder_epochs=50, autoencoder_batch_size=100, model_epochs=30, model_batch_size=10, verbose=verbose)
    net.train(noisy_train, autoencoder_epochs=1, autoencoder_batch_size=100, model_epochs=1, model_batch_size=10,
              verbose=verbose)  # DEBUGGING
    pred_train = net.predict(true_train[:, 0, :])
    pred_test = net.predict(test[:, 0, :])
    train_loss = tf.reduce_mean(tf.keras.losses.MSE(true_train, pred_train)).numpy()
    test_loss = tf.reduce_mean(tf.keras.losses.MSE(test, pred_test)).numpy()

    # TODO: store loss per epoch
    data = {
        "A": lds.A.tolist(),
        "W": W.tolist(),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss)
    }

    with open(filename, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def main():
    intrinsic_dims = [2, 3]
    input_dims = [4, 5]
    num_trials = 2  # num trials per every unique (intrinsic dim, input dim) combo

    print(num_trials)

    # args = []
    # for intrinsic_dim in intrinsic_dims:
    #     for input_dim in input_dims:
    #         filenames = make_filenames(intrinsic_dim, input_dim, num_trials)
    #         args += [[intrinsic_dim, input_dim, filename, False] for filename in filenames]
    #
    # args[0][-1] = True  # turn on logging for the very first element
    #
    # num_processes = len(args)
    # with Pool(num_processes) as p:
    #     p.starmap(do_trial, args)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow logging (only 'FATAL' errors)
    main()
