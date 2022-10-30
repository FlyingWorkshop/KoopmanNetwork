import numpy as np
from numpy.random import default_rng
from scipy.linalg import expm


RNG = default_rng()
TIMESTEPS_PER_TRAJECTORY = 50
TIME = np.linspace(0, 1, TIMESTEPS_PER_TRAJECTORY)


class LDS:
    def __init__(self, A, init_conds, noise_var=0.1):
        """
        Linear first order system of homogeneous equations--i.e., a system of the form x' = Ax
        :param A: real-valued square coeffecient matrix
        :param init_conds: array of initial conditions
        """
        self.A = np.array(A, dtype=np.float64)
        self.dim = self.A.shape[0]
        self.init_conds = np.array(init_conds, dtype=np.float64)
        self.trajs = self.calc_trajs(self.A, self.init_conds, TIME)
        self.noisy_trajs = self.calc_noisy_trajs(self.A, self.init_conds, TIME, noise_var)

    @staticmethod
    def calc_trajs(A, init_conds, time):
        matrix_exponentials = np.array([expm(A * t) for t in time])
        trajs = np.einsum('ijk,lk->lij', matrix_exponentials, init_conds)  # shape = (num trajs, num steps, dim)
        return trajs

    @staticmethod
    def calc_noisy_trajs(A, init_conds, time, noise_var=0.1):
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



