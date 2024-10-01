# Channel Interaction on ODE Datasets for Regular Time Series Forecasting

## Environment Setup

* You can either install the requirements directly using `pip install requirements.txt` into your machine or inside a conda environment.
* Make sure to create a folder `dataset/` in which the datasets will be stored.

## Code and Datasets Description

* The main file for running the experiments is `run_exp.py` which includes all the experiment arguments.
* For every generated dataset from the benchmark, a folder with the model name is created that contains as many .csv files as specified in the generation process (e.g. in our experiments 100) with each file having a timestamp column `t` and as many channel as the number of variables involved in the ODE.

## Running Experiments

* To run experiments, please refer to `scripts/` where we define different example scripts that should run by default 20 Optuna trials to tune the hyperparameters of the defined model for the given dataset.
* For each model [1], [2], and [3], an important argument is passed which is `--channel_independence`, this argument defines if the model is channel independent (--channel_independence 1) or not (--channel_independence 0).
* On each script, you can pass a variety of arguments including `seq_len` and `pred_len` which are 24 and 12 for our experiments.
* Make sure to pass the right number of channels corresponding to the dataset to the argument `--enc_in` so the multivariate models are runnable.
* The hyperparameter tuning range and procedure are defined in the `run_exp.py`.

### Special thanks to [1], as we used their base code as initial starting point for our repo.

[1] Zeng, Ailing, et al. "Are transformers effective for time series forecasting?." Proceedings of the AAAI conference on artificial intelligence. Vol. 37. No. 9. 2023. \
[2] Nie, Yuqi, et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." The Eleventh International Conference on Learning Representations. \
[3] Chen, Si-An, et al. "TSMixer: An All-MLP Architecture for Time Series Forecast-ing." Transactions on Machine Learning Research. 

