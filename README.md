# Physiome-ODE
This is the Git Repository from 
	
Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting based on ODEs,
submitted to the Thirteenth International Conference on Learning Representations (ICLR 2025)

# Requirements
We recommend to create a fresh environment on Python 3.13
```bat
conda create -n physiome python==3.13 
```
```bat
conda activate physiome
```
```bat
pip install -r requirements.txt 
```

# Downloading the dataset

Physiome-ODE can be downloaded on Zenodo: https://zenodo.org/records/11492058

# Recreating or modifying the dataset (Optional)
Instead of downloading our version of Physiome-ODE you can also create your own version with any desired modification. 
For that we shared a detailed step-by-step instruction in the data_creation/README.md. 

# Train models on Physiome-ODE
To run the experiments for different models and datasets use the train_XY.py scripts provided in the experiments/training folder.
As an example we show how to run CRU on the dupont_1991b dataset for fold 0.

```bat
cd  experiments/training
```

```bat
python train_gruodebayes.py -dset dupont_1991b --fold 0
```

To successfully run this you need a folder named dupont_1991b in data/final/. 

# Adding new models to the benchmark. 

Adding new models to Physiome-ODE is simple. We recommend implementing a new model in the training/models folder.
Furthermore, you might consider implementing a custom collate function tailored for your model to map the format 
of the IMTS data from our format to the format that your model. 
As an example see experiments/training/models/train_linodenet.py and its respective
collate function in experiments/training/models/linodenet/utils/data_utils.py

We are open to add new published models to the repo that can be contributed by the community. 