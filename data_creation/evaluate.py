import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# from generate import create

T_range = [0.33, 1, 3.3, 10, 30]
d_c_range = [0.05, 0.1, 0.3]
d_s_range = [0.1, 0.3, 0.5]


def normalize(T, X):
    T_max = np.max(T)
    T = T / T_max
    BS, N_T, dim = X.shape
    std = np.std(X.reshape(BS * N_T, dim), 0)
    mean = np.mean(X.reshape(BS * N_T, dim), 0)
    std[np.argwhere(std == 0)] = 1
    X = (X - mean) / std
    return T, X


def evaluate(args):
    domain = args["domain"]
    model = args["model"]
    files = glob.glob(
        f"raw_data/{domain}/{model}/T={args['T']}_ds={args['d_states']}_dc={args['d_constants']}/*.parquet"
    )
    if len(files) != 100:
        print(f"ERROR: {len(files)} given instead of 100 in raw data for {args}")
        return np.array([-1])
    data = []
    for f in files:
        data.append(pd.read_parquet(f).values)
    # os.remove(f)
    data = np.array(data)
    try:
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
    except:
        return np.array([-1])
    return channel_wise_score


def optimize_dataset(domain, model):
    high_score_mean = -2
    high_score_top_10 = -2
    best_config = {}
    best_config_top10 = {}
    channel_wise_score_for_mean_for_top10 = []
    channel_wise_score_for_mean = []
    for T in T_range:
        for d_c in d_c_range:
            for d_s in d_s_range:
                args = {
                    "domain": domain,
                    "model": model,
                    "total_samples": 100,
                    "T": T,
                    "N_t": 100,
                    "d_constants": d_c,
                    "d_states": d_s,
                    "rand_start": False,
                }
                channel_wise_score = evaluate(args)
                mean_score = np.mean(channel_wise_score)
                score_top10 = np.mean(
                    channel_wise_score[np.argsort(channel_wise_score)[-10:]]
                )
                if mean_score > high_score_mean:
                    high_score_mean = mean_score
                    best_config = args.copy()
                    channel_wise_score_for_mean = channel_wise_score
                if score_top10 > high_score_top_10:
                    high_score_top_10 = score_top10
                    best_config_top10 = args.copy()
                    channel_wise_score_for_mean_for_top10 = channel_wise_score
    result = best_config
    result["high_score"] = high_score_mean
    result["channel_scores"] = [float(i) for i in channel_wise_score_for_mean]
    result_top10 = best_config_top10
    result_top10["high_score"] = high_score_top_10
    result_top10["channel_scores"] = [
        float(i) for i in channel_wise_score_for_mean_for_top10
    ]

    with open(f"models/{domain}/{model}/best_config.json", "w") as outfile:
        json.dump(result, outfile)
    with open(f"models/{domain}/{model}/best_config_top10.json", "w") as outfile:
        json.dump(result_top10, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")

    # Add the arguments
    parser.add_argument(
        "--model", type=str, required=True, help="The path to the file to process"
    )

    parser.add_argument(
        "--domain", type=str, required=True, help="The path to the file to process"
    )

    # Parse the arguments
    args = parser.parse_args()
    print(args)
    optimize_dataset(domain=args.domain, model=args.model)
print("DONE")
