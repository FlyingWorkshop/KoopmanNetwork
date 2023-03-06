import argparse

from numpy.random import default_rng

import koopman as kp

RNG = default_rng()

def main(input_dim, intrinsic_dim, autoencoder_epochs, model_epochs):
    assert intrinsic_dim <= input_dim
    lds = kp.dynamics.RandomLDS(intrinsic_dim)
    W = RNG.integers(-5, 5, size=(intrinsic_dim, input_dim))
    train = lds.make_noisy_trajs(0, 50, 20000, noise_var=0.01) @ W
    in_dist = lds.make_trajs(0, 50, kp.constants.MAX_LINES) @ W
    out_dist = lds.make_trajs(-50, 0, kp.constants.MAX_LINES) @ W
    net = kp.network.KoopmanNetwork(input_dim, intrinsic_dim, model_early_stopping=False)
    net.train(train,
              autoencoder_epochs=autoencoder_epochs,
              autoencoder_batch_size=128,
              model_epochs=model_epochs,
              model_batch_size=128,
              record=[(in_dist, "in-dist"), (out_dist, "out-dist")],
              record_dim=min(intrinsic_dim, 3)
              )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='TrainKoopman',
        description='Trains a Koopman Network and stores the output'
    )
    parser.add_argument('input_dim', type=int)
    parser.add_argument('intrinsic_dim', type=int)
    parser.add_argument('autoencoder_epochs', type=int)
    parser.add_argument('model_epochs', type=int)
    args = parser.parse_args()
    main(args.input_dim, args.intrinsic_dim, args.autoencoder_epochs, args.model_epochs)