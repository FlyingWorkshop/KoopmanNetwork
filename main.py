from pathlib import Path
import json
import multiprocessing
import os

import numpy as np
import tensorflow as tf
from numpy.random import default_rng

import koopman as kp


RNG = default_rng()


def do_trial(intrinsic_dim, input_dim, filename, verbose):
    lds = kp.dynamics.RandomLDS(intrinsic_dim)
    true_train, noisy_train = lds.make_rand_trajs(-50, 50, 5000)
    test = lds.make_rand_trajs(-50, 50, 30, noise_var=0)
    W = RNG.integers(-5, 5, size=(intrinsic_dim, input_dim))
    true_train = np.array([traj @ W for traj in true_train])
    noisy_train = np.array([traj @ W for traj in noisy_train])
    test = np.array([traj @ W for traj in test])
    net = kp.network.KoopmanNetwork(input_dim, intrinsic_dim)
    # net.train(noisy_train, autoencoder_epochs=50, autoencoder_batch_size=100, model_epochs=30, model_batch_size=10, verbose=verbose)
    net.train(noisy_train, autoencoder_epochs=1, autoencoder_batch_size=100, model_epochs=1, model_batch_size=10, verbose=verbose)  # DEBUGGING
    pred_train = net.predict(true_train[:, 0, :])
    pred_test = net.predict(test[:, 0, :])
    train_loss = tf.reduce_mean(tf.keras.losses.MSE(true_train, pred_train)).numpy()
    test_loss = tf.reduce_mean(tf.keras.losses.MSE(test, pred_test)).numpy()
    data = {
        "A": lds.A.tolist(),
        "W": W.tolist(),
        "true_train": true_train[:30].tolist(),
        "noisy_train": true_train[:30].tolist(),
        "test": test.tolist(),
        "pred_train": pred_train.tolist(),
        "pred_test": pred_test.tolist(),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss)
    }
    with open(filename, "w") as f:
        json.dump(data, f, sort_keys=True, indent=4)


def main():
    # TODO: make this command line arguments!
    intrinsic_dim = 2
    input_dim = 4
    num_trials = 2

    # create filenames
    folder = Path(f"cache/{intrinsic_dim}/{input_dim}")
    folder.mkdir(parents=True, exist_ok=True)
    num_files = len(list(folder.glob('**/*.json')))
    filenames = [f"cache/{intrinsic_dim}/{input_dim}/{i}.json" for i in range(num_files + 1, num_files + 1 + num_trials)]
    args = [(intrinsic_dim, input_dim, filename, 0) for filename in filenames[:-1]]
    args.append( (intrinsic_dim, input_dim, filenames[-1], "auto") )  # allow logging on the last network

    num_pools = min(multiprocessing.cpu_count(), num_trials)
    with multiprocessing.Pool(num_pools) as p:
        p.starmap(do_trial, args)


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow logging (only 'FATAL' errors)
    main()
