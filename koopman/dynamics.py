import numpy as np
from numpy.random import default_rng
from scipy.linalg import expm

from .constants import TIME

RNG = default_rng()


class LDS:
    def __init__(self, A):
        """
        Linear first order system of homogeneous equations--i.e., a system of the form x' = Ax
        :param A: real-valued square coeffecient matrix
        """
        self.A = np.array(A, dtype=np.float64)
        self.dim = self.A.shape[0]

    def make_trajs(self, low, high, num_trajs):
        init_conds = RNG.integers(low, high, size=(num_trajs, self.dim))
        return self._calc_trajs(self.A, init_conds, TIME)

    def make_noisy_trajs(self, low, high, num_trajs, noise_var=0.01):
        init_conds = RNG.integers(low, high, size=(num_trajs, self.dim))
        noisy_trajs = self._calc_noisy_trajs(self.A, init_conds, TIME, noise_var)
        return noisy_trajs

    @staticmethod
    def _calc_trajs(A, init_conds, time=TIME):
        matrix_exponentials = np.array([expm(A * t) for t in time])
        trajs = np.einsum('ijk,lk->lij', matrix_exponentials, init_conds)  # shape = (num trajs, num steps, dim)
        return trajs

    @staticmethod
    def _calc_noisy_trajs(A, init_conds, time=TIME, noise_var=0.1):
        """
        Generate noisy trajectories. Noise is added DURING trajectory generation rather than as a post-processing step.
        """
        dt = time[1] - time[0]
        matrix_exponential = expm(A * dt)
        noisy_trajs = []
        for x0 in init_conds:
            traj = [x0]
            for _ in time[1:]:
                noise = RNG.normal(0, noise_var ** 2, size=(x0.shape))
                traj.append(matrix_exponential @ traj[-1] + noise)
            noisy_trajs.append(traj)
        noisy_trajs = np.array(noisy_trajs)
        return noisy_trajs


class RandomLDS(LDS):
    def __init__(self, dim: int, low=-5, high=5):
        A = RNG.integers(low, high, size=(dim, dim))
        super().__init__(A)

