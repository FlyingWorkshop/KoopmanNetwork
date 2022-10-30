from numpy.random import default_rng
import numpy as np
from tqdm.auto import tqdm

from dynamics import LDS
from koopman import KoopmanNetwork
import utils

RNG = default_rng()
W_BOUNDS = (-100, 100)


def do_workout(lds, input_dims, num_reps):
    for input_dim in input_dims:
        do_set(lds, input_dim, num_reps)


def do_set(lds, input_dim, num_reps):
    for W in RNG.integers(W_BOUNDS[0], W_BOUNDS[1], size=(num_reps, lds.dim, input_dim)):
        trans_trajs = np.array([traj @ W for traj in lds.trajs])
        trans_noisy_trajs = np.array([traj @ W for traj in lds.noisy_trajs])
        do_rep()


# TODO: wait till talk w/ Max
def do_rep():
    pass
    # network = KoopmanNetwork(input_dim=input_dim, intrinsic_dim=intrinsic_dim,
    #                          encoder_hidden_widths=(10, 10),
    #                          decoder_hidden_widths=(10, 10), activation="relu")
    # TODO: finish evaluation pipeline


def main():
    intrinsic_dim = 2
    A = RNG.integers(low=-4, high=4, size=(intrinsic_dim, intrinsic_dim))
    init_conds = RNG.integers(low=-100, high=100, size=(5000, intrinsic_dim))
    lds = LDS(A, init_conds, noise_var=0.1)
    do_workout(lds, input_dims=[2, 3, 4, 5], num_reps=10)


if __name__ == '__main__':
    main()