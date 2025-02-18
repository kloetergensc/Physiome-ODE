# Physiome-ODE Data Creation
This directory contains all the steps to create the Physiome-ODE dataset collection.
All these steps can be skipped by simply downloading the dataset from zenodo.
Simply put the folder named "final/" in the "data/" folder, and you can run the training scripts.

## Crawl the ODE models from the Physionet.
The first step to recreating the Physiome-ODE dataset is to collect the generated Python files from the Physiome repository.
To do so, you can execute the crawl.py file. 
Since we cannot guarantee that PhysioMe still has the exact same content as during the time we ran this script, 
re-creating Physiome-ODE from scratch might lead to different datasets.

## Tuning and evaluating the ODE models. 
To curate Physiome-ODE, we first tuned and evaluated models. 
To create 100 time series for each 
possible combination of $\sigma_{dur} (T)$, $\sigma_{states} (d_s)$, $\sigma_{const} (d_c)$ from an ODE model use generate_raw_data.py. E.g. for dupont-1991a execute:


python data_creation/generate_raw_data.py --model dupont_1991b --domain calcium_dynamics


To compute the optimal configuration and the average JDG score of the top 10 channels run 
the evaluate.py script:


python data_creation/evaluate.py --model dupont_1991b --domain calcium_dynamics


## Creating the time series instances for the benchmark

After choosing the ode models and respective you can execute the
generate_benchmark.py, e.g.

python data_creation/generate_benchmark.py --model dupont_1991b --domain calcium_dynamics --T 10 --ds 0.3 --dc 0.05

For that purpose we included the 50 models of IMTS, with respective $\sigma_{dur}$, $\sigma_{states}$, $\sigma_{const}$ in resources/Top50_final.csv . If you want to create this yourself you can use the code in 

notebooks/get_best.ipynb 


Note that the Top50_final.csv is necassary to run the next step as well.

## Transforming the time series instances into IMTS
To transform the 2000 instances created in the previous step into 5 folds of IMTS
data with train, validation and test split run the generate_imts.py script:


python data_creation/generate_imts.py --model dupont_1991b
