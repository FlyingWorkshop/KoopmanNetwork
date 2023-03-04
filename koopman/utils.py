import random
import string

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from sklearn.decomposition import PCA

from .constants import TIMESTEPS_PER_TRAJECTORY, MAX_LINES


def _plot2d(trajs_grid, labels: list = None):
    fig = plt.figure()
    ax = fig.add_subplot()
    for i, trajs in enumerate(trajs_grid):
        if labels is None:
            line_collection = LineCollection(trajs, color=f"C{i}")
        else:
            line_collection = LineCollection(trajs, color=f"C{i}", label=labels[i])
        ax.add_collection(line_collection)
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    if labels is not None:
        ax.legend()
    return fig, ax


def _plot3d(trajs_grid, labels: list = None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i, trajs in enumerate(trajs_grid):
        if labels is None:
            line_collection = Line3DCollection(trajs, color=f"C{i}")
        else:
            line_collection = Line3DCollection(trajs, color=f"C{i}", label=labels[i])
        ax.add_collection(line_collection)
    minima = trajs_grid.min(axis=(0, 1, 2))
    maxima = trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    ax.set_zlim(minima[2], maxima[2])
    if labels is not None:
        ax.legend()
    return fig, ax


def _plot4d(trajs_grid, pca: PCA, labels: list = None):
    assert pca.n_components == 2 or pca.n_components == 3
    trajs_grid = _apply_pca_to_grid(trajs_grid, pca)
    if pca.n_components == 2:
        return _plot2d(trajs_grid, labels)
    else:
        return _plot3d(trajs_grid, labels)


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


def plot(trajs_grid: list, target_dim=2, pca=None, max_lines=MAX_LINES, labels: list = None):
    trajs_grid = np.array([elem[:max_lines] for elem in trajs_grid])
    dim = trajs_grid.shape[-1]
    assert target_dim <= dim
    assert target_dim <= 3
    if dim == 2:
        return _plot2d(trajs_grid, labels)
    elif dim == 3 and target_dim == 3:
        return _plot3d(trajs_grid, labels)
    else:
        pca = pca or make_pca(trajs_grid, n_components=target_dim)
        return _plot4d(trajs_grid, pca, labels)


def _rand_alphanumeric():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))


def load_recording(filename, max_lines=MAX_LINES):
    recording = pd.read_csv(filename)
    epochs = len(recording)
    trajs = np.array([np.fromstring(s, sep=",", dtype=float) for s in recording['trajs']])
    trajs = trajs.reshape((epochs, max_lines, TIMESTEPS_PER_TRAJECTORY, -1))
    return trajs


def _animate2d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels):
    # create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot()
    minima = static_trajs_grid.min(axis=(0, 1, 2))
    maxima = static_trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])

    static_artists = []
    for i, trajs in enumerate(static_trajs_grid):
        line_collection = LineCollection(trajs, color=f"C{i}", alpha=0.5, lw=1, label=static_labels[i])
        artist = ax.add_collection(line_collection)
        static_artists.append(artist)

    animated_artists = []
    for epoch in range(animated_trajs_grids.shape[1]):
        artists = static_artists.copy()
        for i, trajs_grid in enumerate(animated_trajs_grids):
            color = f"C{len(static_trajs_grid) + i}"
            label = animated_labels[i] if epoch == 0 else None
            line_collection = LineCollection(trajs_grid[epoch], color=color, label=label)
            artist = ax.add_collection(line_collection)
            artists.append(artist)
        animated_artists.append(artists)
    ax.legend()
    recording = animation.ArtistAnimation(fig, animated_artists, interval=40, blit=True)
    return recording


def _animate3d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels):
    # create figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    minima = static_trajs_grid.min(axis=(0, 1, 2))
    maxima = static_trajs_grid.max(axis=(0, 1, 2))
    ax.set_xlim(minima[0], maxima[0])
    ax.set_ylim(minima[1], maxima[1])
    ax.set_zlim(minima[2], maxima[2])

    static_artists = []
    for i, trajs in enumerate(static_trajs_grid):
        line_collection = Line3DCollection(trajs, color=f"C{i}", alpha=0.5, lw=1, label=static_labels[i])
        artist = ax.add_collection(line_collection)
        static_artists.append(artist)

    animated_artists = []
    for epoch in range(animated_trajs_grids.shape[1]):
        artists = static_artists.copy()
        for i, trajs_grid in enumerate(animated_trajs_grids):
            color = f"C{len(static_trajs_grid) + i}"
            label = animated_labels[i] if epoch == 0 else None
            line_collection = Line3DCollection(trajs_grid[epoch], color=color, label=label)
            artist = ax.add_collection(line_collection)
            artists.append(artist)
        animated_artists.append(artists)
    ax.legend()
    recording = animation.ArtistAnimation(fig, animated_artists, interval=40, blit=True)
    return recording


def _animate4d(animated_trajs_grids,
               animated_labels,
               static_trajs_grid,
               static_labels,
               pca: PCA):
    assert pca.n_components == 2 or pca.n_components == 3
    animated_trajs_grids = _apply_pca_to_grid(animated_trajs_grids, pca)
    static_trajs_grid = _apply_pca_to_grid(static_trajs_grid, pca)
    if pca.n_components == 2:
        return _animate2d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels)
    else:
        return _animate3d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels)


def animate(animated_trajs_grids: list,
            animated_labels: list,
            static_trajs_grid: list,
            static_labels: list,
            target_dim=2,
            pca=None,
            max_lines=MAX_LINES):
    # clip trajectories
    animated_trajs_grids = np.array(animated_trajs_grids)[:, :, :max_lines]
    static_trajs_grid = np.array([elem[:max_lines] for elem in static_trajs_grid])
    dim = static_trajs_grid.shape[-1]
    assert target_dim <= dim
    assert target_dim <= 3
    if dim == 2:
        return _animate2d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels)
    elif dim == 3 and target_dim == 3:
        return _animate3d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels)
    else:
        pca = pca or make_pca(static_trajs_grid, n_components=target_dim)
        return _animate4d(animated_trajs_grids, animated_labels, static_trajs_grid, static_labels, pca)


def make_recording(tag, target_dim=2, pca=None, max_lines=MAX_LINES):
    in_dist_gold = np.load(f"cache/{tag}-in-dist-gold.npy")
    out_dist_gold = np.load(f"cache/{tag}-out-dist-gold.npy")
    in_dist_preds = load_recording(f"cache/{tag}-in-dist-preds.csv")
    out_dist_preds = load_recording(f"cache/{tag}-out-dist-preds.csv")
    recording = animate(animated_trajs_grids=[in_dist_preds, out_dist_preds],
                        animated_labels=["in dist. pred", "out dist. pred"],
                        static_trajs_grid=[in_dist_gold, out_dist_gold],
                        static_labels=["in dist. gold", "out dist. gold"],
                        target_dim=target_dim,
                        pca=pca,
                        max_lines=max_lines)
    return recording
