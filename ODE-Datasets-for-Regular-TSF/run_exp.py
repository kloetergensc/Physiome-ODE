import argparse
import os
import random
import numpy as np
import torch
from exp.exp_main import Exp_Main
import optuna
import logging
import sys
import uuid
from datetime import datetime
import traceback

fix_seed = 3000
random.seed(fix_seed)
torch.manual_seed(fix_seed)
torch.autograd.set_detect_anomaly(True)
np.random.seed(fix_seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def int_list(s):
    # If the input is already a list, return it
    if isinstance(s, list):
        return s
    # If the input is a string, split it by commas and convert to a list of integers
    return list(map(int, s.split(',')))

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
parser.add_argument('--model', type=str, required=False, default='TSMixer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=False, default='ode', help='dataset type')
parser.add_argument('--root_path', type=str, default='dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electrophysiology/inada_N_2009', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=24, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')

# DLinear
# parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')


# Mixers
parser.add_argument('--num_blocks', type=int, default=3, help='number of mixer blocks to be used in TSMixer')
parser.add_argument('--hidden_size', type=int, default=32, help='first dense layer diminsions for mlp features block')
parser.add_argument('--single_layer_mixer', type=str2bool, nargs='?', default=False, help="if true a single layer mixers are used")

#Common between Mixers and Transformer-based models
parser.add_argument('--activation', type=str, choices={"gelu", "relu", "linear"}, default='gelu', help='activation')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
parser.add_argument('--early_stopping', type=str2bool, nargs='?',
                        const=True, default=True,
                        help="whether to include early stopping or not")
parser.add_argument('--enc_in', type=int, default=29, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--norm', type=str, choices={"batch", "instance"}, default="batch", help="type of normalization")

# Patching and Convolution models
parser.add_argument('--excluded_component', type=int, default=0, help="Number of component to excluded from mixing in PatchTSMixer model; 0 : all included, 1 : intra patch mixing, 2 : inter patch mixing, 3 : inter channel mixing")
parser.add_argument('--patch_size', type=int, default=16, help="Number of timesteps per patch")
# parser.add_argument('--kernel_size', type=int, default=1, help="conv width to cover certain number of timesteps")
# parser.add_argument('--stride', type=int, default=1, help='number of non-overlapping timesteps for conv operation')
# parser.add_argument('--affine', type=str2bool, default=True, help='define if the rev_norm is affine or not')
parser.add_argument('--embedding_dim', type=int, default=128, help="Embedding dimension for models including patch embedding")
parser.add_argument('--channel_kernel', type=int, default=2, help="Decides the number of channels being mixed at each convolution step")
parser.add_argument('--channel_stride', type=int, default=1, help="Decides the number of non-overlapping channels during the mixing process")
parser.add_argument('--channel_dilation', type=int, default=1, help="Decides the dilation between the channels being mixed")
parser.add_argument('--temporal_kernel_size', type=int, default=2, help="Number of tokens being mixed")
parser.add_argument('--temporal_stride', type=int, default=1, help="number of non overlapping tokens during the mixing process")
parser.add_argument('--temporal_dilation', type=int, default=1, help="Dilation applied on the temporal dimension (number of patches dimension)")

# patchtst
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--channel_independence', type=int, default=0, help="independent channel")


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument("--use_gpu", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="use gpu")
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Optuna Parameters
parser.add_argument("--trials", type=int, default=20)
parser.add_argument("--use_optuna", type=str2bool, default=True)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

class Objective(object):
    def __init__(self, args):
        self.args = args

    def __call__(self, trial):
        # setting record of experiments
        global best_model_args
        global best_model_settings
        global best_model
        args.unique_id = str(uuid.uuid4())
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.dataset_name,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        args.channel_independence,
        args.unique_id, 
          '0')

        if args.model=="PatchTST":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
                "e_layers" : trial.suggest_int("e_layers", 1.0, 6.0),
                "d_ff" : trial.suggest_categorical("d_ff", [256, 512, 1024] ),
                "d_model" : trial.suggest_categorical("d_model", [128, 256, 512, 1024]),
                "dropout" : trial.suggest_float("dropout", 0, 0.9),
                "fc_dropout" : trial.suggest_float("fc_dropout", 0, 0.9),
                "patch_size" : trial.suggest_categorical("patch_size", [4, 8]),
                "stride" : trial.suggest_categorical("stride", [2, 4]),
                "unique_id": args.unique_id
                }
        elif args.model == "TSMixer":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
                "num_blocks" : trial.suggest_int("num_blocks", 1.0, 5.0),
                "hidden_size" : trial.suggest_categorical("hidden_size", [32, 64, 256, 1024]),
                "dropout" : trial.suggest_float("dropout", 0, 0.9),
                "activation": trial.suggest_categorical("activation", ["gelu", "relu"]),
                "unique_id": args.unique_id 

                }
        elif args.model == "DLinear":
            params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1e-2, log=True),
                "unique_id" : args.unique_id 

            }
        
        vars(args).update(params)
        print("args :",  args)
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        try:
            _, vali_loss = exp.train(setting) # get the validation loss
            if trial.number == 0:
                # Keeping track of the args and setting of the best model
                best_model = exp.model
                best_model_args = args 
                best_model_settings = setting
                print("---------- Best Model Updated -------------")
            elif  vali_loss < study.best_value :
                # Keeping track of the args and setting of the best model
                best_model = exp.model
                best_model_args = args 
                best_model_settings = setting
                print("---------- Best Model Updated -------------")

            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)
            torch.cuda.empty_cache()

        except Exception as e:
            print("error !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"An error occurred: {e}")
            traceback.print_exc()
            vali_loss = 10
            torch.cuda.empty_cache()
        
        return vali_loss
        
optuna_=args.use_optuna


if optuna_:
    sampler = optuna.samplers.TPESampler(seed=100)
    dataset_path_array = args.data_path.split("/")
    args.dataset_name = dataset_path_array[len(dataset_path_array)-2] + "_" + dataset_path_array[len(dataset_path_array)-1]
    study_name = args.model + "_" + args.dataset_name + "_"+ str(args.pred_len) + "_"+ str(args.channel_independence) + "_date_" + str(datetime.now())
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(direction="minimize",
                            study_name=study_name,
                            storage=storage_name,
                            sampler=sampler,
                            load_if_exists=True)
    optuna_objective = Objective(args)
    study.optimize(optuna_objective, n_trials=args.trials)

    # Printing the parameters for the best performing model
    print("Best hyperparameters: ", study.best_params)
    print("Best loss: ", study.best_value)

    # Evaluating the model on the test dataset
    if not args.train_only:
        exp = Exp(best_model_args)
        print('>>>>>>>testing with best hyperparameters : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(study.best_params))
        exp.test(best_model_settings, test=1, model=best_model) # test=1 is critical to load the model according to the best settings


else:
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            _, vali_loss = exp.train(setting) # get the validation loss

            if not args.train_only:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                    args.model,
                                                                                                    args.data,
                                                                                                    args.features,
                                                                                                    args.seq_len,
                                                                                                    args.label_len,
                                                                                                    args.pred_len,
                                                                                                    args.d_model,
                                                                                                    args.n_heads,
                                                                                                    args.e_layers,
                                                                                                    args.d_layers,
                                                                                                    args.d_ff,
                                                                                                    args.factor,
                                                                                                    args.embed,
                                                                                                    args.distil,
                                                                                                    args.des, ii)

        exp = Exp(args)  # set experiments
        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        else:
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, test=1)
        torch.cuda.empty_cache()
