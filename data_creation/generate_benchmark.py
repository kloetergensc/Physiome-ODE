import argparse

import pandas as pd
from generate import create

parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument("--model", type=str, required=True, help="The name of the model")
parser.add_argument("--domain", type=str, required=True, help="The domain of the model")
parser.add_argument("--T", type=float, required=True, help="sigma_dur")
parser.add_argument("--dc", type=float, required=True, help="\sigma_const")
parser.add_argument("--ds", type=float, required=True, help="\sigma_states")


# Parse the arguments
args = parser.parse_args()
print(args)

create(
    {
        "domain": args.domain,
        "model": args.model,
        "total_samples": 2000,
        "T": args.T / 2,
        "N_t": 100,
        "d_constants": args.dc,
        "d_states": args.ds,
        "rand_start": True,
        "final": True,
    }
)
print("DONE")
