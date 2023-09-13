import torch
import numpy as np
import copy
import random
import argparse
from datetime import datetime

# reproducibility
def seed(seed):
    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

# arguement parser
def Args(FL):
    parser = argparse.ArgumentParser()
    
    # general parameters for both non-FL and FL
    parser.add_argument('-p', '--project', type = str, default = 'korea', help = 'project name, from colorado, korea, daisee, engagenet')
    parser.add_argument('--name', type = str, default = 'name', help = 'wandb run name')
    parser.add_argument('-seed', '--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('-fl_csv', '--fl_csv', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use FL data split or non-FL data split')
    parser.add_argument('-mg', '--use_meglass', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to use meglass feature')
    parser.add_argument('--h_size', type = int, default = 100, help = 'hidden size of LSTM model')
    parser.add_argument('--n_layer', type = int, default = 3, help = 'hidden layers of LSTM model')
    parser.add_argument('-sq', '--seq_length', type = int, default = 124, help = 'data sequence length')
    parser.add_argument('-bs', '--batch_size', type = int, default = 4, help = 'batch size')
    parser.add_argument('-bd', '--bi_dir', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use bidirectional LSTM')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = -3, help = 'client learning rate in exponent')
    parser.add_argument('--global_epoch', type = int, default = 401, help = 'number of global aggregation rounds')
    parser.add_argument('-c_op', '--client_optim', default = torch.optim.SGD, help = 'client optimizer')
                    
    # general parameters for FL
    parser.add_argument('-fl', '--switch_FL', type = str, default = 'FedAvg', help = 'FL algorithm, from FedAvg, FedAdam, FedProx, MOON, FedAwS, TurboSVM')
    parser.add_argument('-C', '--client_C', type = float, default = 0.5, help = 'number of participating clients in each aggregation round')
    parser.add_argument('-E', '--client_epoch', type = int, default = 8, help = 'number of client local training epochs')
    
    # for FedOpt (FedAdam)
    parser.add_argument('-g_lr', '--global_lr', type = float, default = -3, help = 'global learning rate in exponent')
    parser.add_argument('-g_op', '--global_optim', default = torch.optim.Adam, help = 'global optimizer')
    
    # for FedAwS and TurboSVM
    parser.add_argument('-l_lr', '--logits_lr', type = float, default = -3, help = 'global learning rate for logit layer in exponent')
    parser.add_argument('-l_op', '--logits_optim', default = torch.optim.Adam, help = 'global optimizer for logit layer')
    
    args = parser.parse_args()
    args.time = str(datetime.now())[5:-10]
    args.fed_agg = None
    args.MOON = False
    args.FedProx = False

    # reuse optimizer or not
    args.FL = FL
    args.reuse_optim = not FL
    
    # features
    args.which_feature = ['emonet', 'openface_8', 'meglass'] if args.use_meglass else ['emonet', 'openface_8']

    # paths
    args.paths = {
            'colorado': {
                        'data_split_csv_path_FL' : './datasets/dataset_summary_sidney_random_FL.csv',
                        'data_split_csv_path_nFL': './datasets/dataset_summary_sidney_random.csv'   , 
                        'data_csv_path'          : '../MW_vids_Sidney/fold_ids_reduced.csv'         ,
                        'emonet_path'            : '../MW_vids_Sidney/emonet/'                      ,
                        'openface_path'          : '../MW_vids_Sidney/OpenFace2.2.0/'               ,
                        'meglass_path'           : '../MW_vids_Sidney/meglass/'                     ,
                        },
            'korea'   : {
                        'data_split_csv_path_FL' : './datasets/dataset_summary_korea_random_FL.csv'                 ,
                        'data_split_csv_path_nFL': './datasets/dataset_summary_korea_random.csv'                    ,
                        'data_csv_path'          : '../Mind_Wandering_Detection_Data_Korea/fold_ids.csv'            ,
                        'emonet_path'            : '../Mind_Wandering_Detection_Data_Korea/features/emonet/'        ,
                        'openface_path'          : '../Mind_Wandering_Detection_Data_Korea/features/OpenFace2.2.0/' ,
                        'meglass_path'           : '../Mind_Wandering_Detection_Data_Korea/features/meglass/'       ,
                        },
            'daisee'  : {
                        'data_split_csv_path_FL' : './datasets/dataset_summary_DAiSEE_boredom.csv',
                        'data_split_csv_path_nFL': './datasets/dataset_summary_DAiSEE_boredom.csv',
                        'data_csv_path'          : '../DAiSEE/fold_ids.csv'                       ,
                        'emonet_path'            : '../DAiSEE/emonet/'                            ,
                        'openface_path'          : '../DAiSEE/Openface/'                          ,
                        'meglass_path'           : '../DAiSEE/meglass/'                           ,
                        },
            'engagenet':{
                        'data_split_csv_path_FL' : './datasets/dataset_summary_EngageNet.csv',
                        'data_split_csv_path_nFL': './datasets/dataset_summary_EngageNet.csv',
                        'data_csv_path'          : '../EngageNet/fold_ids.csv'               ,
                        'emonet_path'            : '../EngageNet/emonet/'                    ,
                        'openface_path'          : '../EngageNet/Openface/'                  ,
                        'meglass_path'           : '../EngageNet/meglass/'                   ,
                        }
        }

    return args
        
def switch_FL(args):
    match args.switch_FL:

        case 'FedAvg':
            args.fed_agg = 'FedAvg'

        case 'FedAdam':
            args.fed_agg = 'FedOpt'
            
        case 'FedProx':
            args.fed_agg = 'FedAvg'
            args.FedProx = True

        case 'MOON':
            args.fed_agg = 'FedAvg'
            args.MOON = True

        case 'FedAwS':
            args.fed_agg = 'FedAwS'
            
        case 'TurboSVM':
            args.fed_agg = 'TurboSVM'
            
        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)

# get weights for classes 0 and 1
def get_imbalance_weight(labels):
    unique_labels, unique_counts = labels.unique(return_counts = True, sorted = True)
    assert(unique_labels.equal(torch.tensor([0, 1])))
    sum_counts = sum(unique_counts)
    weight = torch.tensor([unique_counts[1] / sum_counts, unique_counts[0] / sum_counts])
    return weight

# average model parameters
def weighted_avg_params(params, weights = None):
    if weights == None:
        weights = [1.0] * len(params)
        
    params_avg = copy.deepcopy(params[0])
    for key in params_avg.keys():
        params_avg[key] *= weights[0]
        for i in range(1, len(params)):
            params_avg[key] += params[i][key] * weights[i]
        params_avg[key] = torch.div(params_avg[key], sum(weights))
    return params_avg

# compute weighted average
def weighted_avg(values, weights):
    sum_values = 0
    for v, w in zip(values, weights):
        sum_values += v *w
    return sum_values / sum(weights)

# get weight vector
# w: weight per class
# output : weight per sample
def weight_to_vec(w, y):
    w = w.to(y.device)
    wv = w[y.long()]
    return wv