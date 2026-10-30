# Physiome-ODE
This is the Git repository of

[Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time-Series Forecasting Based on Biological ODEs](https://proceedings.iclr.cc/paper_files/paper/2025/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html)
(Klötergens et al., ICLR 2025)

# Leaderboard

The current leaderboard, results per dataset, samples of all 50 datasets and the guide for submitting a model
are on the website: **https://kloetergensc.github.io/Physiome-ODE/**

The leaderboard has grown since the Physiome-ODE paper, so its rankings differ from the paper's tables.
The per-dataset results of the models from the paper are unchanged.

- **October 2026:** Added TIDES, with results from Table 8 of
  [TIDES: Implicit Time-Awareness in Selective State Space Models](https://arxiv.org/abs/2605.09742)
  (Soydan et al., NeurIPS 2026); its code is in `experiments/training/models/tides`. TIDES counts as
  ODE-based: its latent state follows a linear ODE solved in closed form, as in LinODENet and CRU. The site
  now compares five ODE-based models by default and ten in total (LinODENet: 23 wins and average rank 1.86
  among the ODE-based models, 5 and 3.86 among all ten; TIDES: 26 and 1.64, 8 and 3.42).
- **October 2026:** Added IMTS-Mixer, TimeCHEAT and tPatchGNN, with results from Table 6 of
  [Mixing It Up: Exploring Mixer Networks for Irregular Multivariate Time Series Forecasting](https://arxiv.org/abs/2502.11816)
  (Klötergens et al., PAKDD 2026); the Physiome-ODE results are only in the arXiv version. The site
  compares the four ODE-based models by default; with "All IMTS models" selected, wins and average
  ranks are computed over all nine. Either way they differ from the Physiome-ODE paper, which compared
  six models (LinODENet: 25 wins and average rank 2.10 in the paper, 34 and 1.36 among the ODE-based models,
  7 and 3.36 among all nine).


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

# Citation

If you use Physiome-ODE or the leaderboard, please cite the
[ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html):

```bibtex
@inproceedings{klotergens2025physiomeode,
  title     = {{Physiome-ODE}: A Benchmark for Irregularly Sampled Multivariate
               Time-Series Forecasting Based on Biological {ODEs}},
  author    = {Kl{\"o}tergens, Christian and Yalavarthi, Vijaya Krishna and
               Scholz, Randolf and Stubbemann, Maximilian and Born, Stefan and
               Schmidt-Thieme, Lars},
  booktitle = {International Conference on Learning Representations (ICLR)},
  volume    = {2025},
  pages     = {34961--34988},
  year      = {2025},
  url       = {https://proceedings.iclr.cc/paper_files/paper/2025/hash/565f995643da6329cec701f26f8579f5-Abstract-Conference.html}
}
```
