import argparse

import pandas as pd
from generate import create

T_range = [0.33, 1, 3.3, 10, 30]
d_c_range = [0.05, 0.1, 0.3]
d_s_range = [0.1, 0.3, 0.5]

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
for T in T_range:
    for d_c in d_c_range:
        for d_s in d_s_range:
            try:
                create(
                    {
                        "domain": args.domain,
                        "model": args.model,
                        "total_samples": 100,
                        "T": T,
                        "N_t": 100,
                        "d_constants": d_c,
                        "d_states": d_s,
                        "rand_start": False,
                        "final": False,
                    }
                )
            except:
                print(T, d_c, d_s, "not feasable")
    print("DONE")
