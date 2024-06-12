import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# https://matplotlib.org/stable/gallery/mplot3d/lorenz_attractor.html

OUT_PATH = "data/raw_data/lorenz"
N = 200
T = 30
steps = 200
delta_t = T / steps


def lorenz(xyz, t, s=10, r=28, b=2.667):
    """
    Parameters
    ----------
    xyz : array-like, shape (3,)
       Point of interest in three-dimensional space.
    s, r, b : float
       Parameters defining the Lorenz attractor.

    Returns
    -------
    xyz_dot : array, shape (3,)
       Values of the Lorenz attractor's partial derivatives at *xyz*.
    """
    x, y, z = xyz
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def create_and_save(i):
    np.random.seed(i)
    x = np.random.uniform(1, 3)
    y = np.random.uniform(0, 2)
    z = np.random.uniform(0, 2)
    # x, y, z = 2, 1, 1
    xyz = x, y, z
    model = [xyz]
    for step in range(steps):
        t_int = [0, delta_t]
        sol = odeint(lorenz, xyz, t_int)
        model.append(sol[1])
        xyz = sol[1]
    model = np.array(model)
    start = np.random.randint(0, 100)
    model = model[start : start + 100 :,]
    model = pd.DataFrame(data=model, columns=["X", "Y", "Z"])
    times = pd.DataFrame({"t": np.linspace(0, T, 100)})
    model = pd.concat([times, model], axis=1)
    model.to_parquet(f"{OUT_PATH}/ts-data-{str(i)}.parquet", index=None)


def generate():
    total_processes = cpu_count()
    if not os.path.isdir(OUT_PATH):
        os.mkdir(OUT_PATH)
    iters = range(N)
    with Pool(processes=total_processes) as pool:
        pool.map(create_and_save, iters)


if __name__ == "__main__":
    generate()
