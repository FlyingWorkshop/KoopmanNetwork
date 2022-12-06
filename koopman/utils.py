import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA


def _plot2d(trajs_grid, labels: list[str] = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, trajs in enumerate(trajs_grid):
        if labels is None:
            lines = LineCollection(trajs, color=f"C{i}")
        else:
            lines = LineCollection(trajs, color=f"C{i}", label=labels[i])
        ax.add_collection(lines)
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    if labels is not None:
        ax.legend()
    plt.show()


def _plot3d(trajs_grid, labels: list[str] = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, trajs in enumerate(trajs_grid):
        if labels is None:
            lines = LineCollection(trajs, color=f"C{i}")
        else:
            lines = Line3DCollection(trajs, color=f"C{i}", label=labels[i])
        ax.add_collection(lines)
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    ax.set_zlim(minima[2], maxima[2])
    if labels is not None:
        ax.legend()
    plt.show()


def _plot4d(trajs_grid, pca: PCA, labels: list[str] = None):
    assert pca.n_components == 2 or pca.n_components == 3
    trajs_grid = _apply_pca_to_grid(trajs_grid, pca)
    if pca.n_components == 2:
        _plot2d(trajs_grid, labels)
    else:
        _plot3d(trajs_grid, labels)


def make_pca(trajs: np.ndarray, n_components=3):
    dim = trajs.shape[-1]
    X = trajs.reshape((-1, dim))
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def _apply_pca_to_grid(trajs_grid, pca) -> np.ndarray:
    dim = trajs_grid.shape[-1]
    X = trajs_grid.reshape((-1, dim))
    result = pca.transform(X)
    target_shape = list(trajs_grid.shape)
    target_shape[-1] = pca.n_components
    result = result.reshape(target_shape)
    return result


def plot(trajs_grid: list[np.ndarray], target_dim=3, pca=None, max_lines=30, labels: list[str] = None):
    trajs_grid = np.array([elem[:max_lines] for elem in trajs_grid])
    trajs_grid = trajs_grid[..., :max_lines]
    dim = trajs_grid.shape[-1]
    if dim == 2:
        _plot2d(trajs_grid, labels)
    elif dim == 3:
        _plot3d(trajs_grid, labels)
    else:
        if pca is None:
            pca = make_pca(trajs_grid[0], n_components=target_dim)
        _plot4d(trajs_grid, pca, labels)
