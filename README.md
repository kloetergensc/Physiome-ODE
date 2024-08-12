# IMTSBench
This is the Git Repository from 
	
IMTSBench: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting based on ODEs (Kloetergens et. al, 2024),
submitted to The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

# Downloading the dataset

IMTSBench can be downloaded here: https://zenodo.org/records/11492058
The downloaded folder should be copied to data/IMTS_benchmark_datasets/ and renamed to imts_bench

# Recreating or modifying the dataset (Optional)
Instead of downloading our version of IMTS you can also create your own version with any desired modification. 
For that we shared a detailed step-by-step instruction in the data_creation/README.md. 

# Train models on IMTSBench
To run the experiments for different models and datasets use the train_XY.py scripts provided in the experiments/training folder.
As an example we show how to run CRU on the dupont_1991b dataset for fold 0.

'''
python experiments/training/train_gruodebayes.py -dset dupont_1991b --fold 0
'''

To successfully run this you need a folder in data/IMTS_benchmark_datasets. 

# Adding new models to the benchmark. 

Adding new models to IMTSBench is simple. We recommend implementing a new model in the training/models folder.
Furthermore, you might consider implementing a custom collate function tailored for your model to map the format 
of the IMTS data from our format to the format that your model. As an example see experiments/training/models/train_linodenet.py and its respective
collate function in experiments/training/models/linodenet/utils/data_utils.py
