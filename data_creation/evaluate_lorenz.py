import glob

import numpy as np
import pandas as pd
from evaluate import normalize


def main():
    files = glob.glob(f"data/raw_data/lorenz/*.parquet")
    data = []
    for f in files:
        data.append(pd.read_parquet(f).values)
    # os.remove(f)
    data = np.array(data)
    # try:
    T = data[:, :, 0]
    X = data[:, :, 1:]
    if np.max(np.absolute(X)) > 1e13:
        return np.array([-1])
    T, X = normalize(T, X)
    if np.max(np.absolute(X)) > 10:
        return np.array([-1])
    avg_std_of_diff_in_channels = np.mean(np.std(np.diff(X, 1, 1), 1), 0)
    avg_std_of_diff_in_samples = np.mean(np.std(np.diff(X, 1, 1), 0), 0)
    channel_wise_score = avg_std_of_diff_in_channels * avg_std_of_diff_in_samples
    # except:
    #    return np.array([-1])
    return channel_wise_score


print(np.mean(main()))
