import argparse
import ast
import csv
import numpy as np
import os
import random
import torch
from time import time

from copy import deepcopy
from functools import partial
from tqdm import tqdm

from losses import get_loss_fn
from IndexTracking import IndexTracking
from data_process import load_target_index
from utils import print_metrics, init_if_not_saved, move_to_gpu

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import wandb

## Set basic directory
os.chdir('/home/hankkim77/experiment_server')

def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation='relu',
    output_activation='sigmoid',
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == 'relu':
            activation_fn = torch.nn.ReLU
        elif activation == 'sigmoid':
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception('Invalid activation function: ' + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1)))
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)), View(num_targets)]

    if output_activation == 'relu':
        net_layers.append(torch.nn.ReLU())
    elif output_activation == 'sigmoid':
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == 'tanh':
        net_layers.append(torch.nn.Tanh())
    elif output_activation == 'softmax':
        net_layers.append(torch.nn.Softmax(dim=-1))

    return torch.nn.Sequential(*net_layers)


def metrics2wandb(
    datasets,
    model,
    problem,
    prefix="",
):
    metrics = {}
    for Xs, Ys, Ys_aux, partition in datasets:
        # Choose whether we should use train or test 
        isTrain = (partition=='train') and (prefix != "Final")
        pred = model(Xs).squeeze()
        Zs_pred = problem.get_decision(pred, aux_data=Ys_aux, isTrain=isTrain)
        objectives = problem.get_objective(Ys, Zs_pred, aux_data=Ys_aux)
        # Print
        objective = objectives.mean().item()
        metrics[partition] = {'objective': objective}
 
    return metrics


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, choices=['knapsack', 'portfolio', 'index'], default='index') 
    parser.add_argument('--loadnew', type=ast.literal_eval, default=True)
    parser.add_argument('--instances', type=int, default=400)
    parser.add_argument('--testinstances', type=int, default=400)
    parser.add_argument('--valfrac', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--intermediatesize', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--earlystopping', type=ast.literal_eval, default=True)
    parser.add_argument('--valfreq', type=int, default=20)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--wandb', type=bool, default=False)
    # parser.add_argument('--twostage', type=bool, default=True)
    #   Learning Losses
    parser.add_argument('--loss', type=str, choices=['gicln', 'icln', 'mse', 'dense', 'dfl', 'quad'], default='dfl')
    parser.add_argument('--serial', type=ast.literal_eval, default=True)
    parser.add_argument('--sampling', type=str, choices=['random', 'random_hessian', 'random_3std'], default='random')
    parser.add_argument('--samplingstd', type=float, default=0.1)
    parser.add_argument('--numsamples', type=int, default=10) # default 1000 / 10
    parser.add_argument('--losslr', type=float, default=0.001)
    #   ICLN-specific: Hyperparameters
    parser.add_argument('--iclnhid', type=int, default=2)
    parser.add_argument('--actfn', type=str, default='SOFTPLUS')
    parser.add_argument('--minmax', type=str, default='MIN')
    # #   ICLN-G-specific: # Samples
    # parser.add_argument('--giclnsmp', type=int, default=10000)
    #   Domain-specific: Portfolio Optimization
    # parser.add_argument('--stocks', type=int, default=50)
    parser.add_argument('--stockalpha', type=float, default=0.1)
    # parser.add_argument('--sampling_std', type=float, default=0.1)
    #   Domain-specific: Index Tracking
    parser.add_argument('--stocks', type=int, default=100) # default 511 
    parser.add_argument('--cardinality', type=float, default=1)  ## default 10
    parser.add_argument('--market',  type=str, choices=['ftse100','nasdaq100','hsi', 'sp500'], default='hsi')  
    parser.add_argument('--save_mode', type=ast.literal_eval, default=True)
    
    parser.add_argument('--exp_num', type=int, default=0)

    parser.add_argument('--dflalpha', type=float, default=10.0) ## default 10


    args = parser.parse_args()

    
    # start a new wandb run to track this script
    wandb.init(
    # set the wandb project where this run will be logged
    project="indextracking_dfl_server_2",

    # track hyperparameters and run metadata
    config={
    "exp_num": args.exp_num,
    "market": args.market,
    "cardinality": args.cardinality,
    "dflalpha": args.dflalpha,
    "loss": args.loss,
    "batch": args.batchsize,
    "layers" : args.layers,
    "learning_rate" :args.lr,
    }
    )
    
    # Load problem 
    print(f"Hyperparameters: {args}\n")
    print(f"Loading {args.problem} Problem...")
    # init_problem = partial(init_if_not_saved, load_new=args.loadnew)  
        

    problem_kwargs =    {'num_train_instances': args.instances,
                        'num_test_instances': args.testinstances,
                        'num_stocks': args.stocks,
                        'K': args.cardinality,
                        'val_frac': args.valfrac,
                        'rand_seed': args.seed,
                        'market': args.market,
                        'save_mode': args.save_mode}
    problem = IndexTracking(**problem_kwargs)
    
    
    print(f"Loading {args.loss} Loss Function...")
    loss_fn = get_loss_fn(
        args.loss,
        problem,
        sampling=args.sampling,
        num_samples=args.numsamples,
        sampling_std=args.samplingstd,
        lr=args.losslr,
        serial=args.serial,
        icln_hidden_num=args.iclnhid,
        icln_actfn=args.actfn,
        minmax=args.minmax,
        dflalpha =args.dflalpha
    )
    
    # Load an ML model to predict the parameters of the problem
    print(f"Building dense Model...")
    ipdim, opdim = problem.get_modelio_shape()
    model = dense_nn(
        num_features=ipdim,
        num_targets=opdim,
        num_layers=args.layers,
        intermediate_size=args.intermediatesize,
        output_activation=problem.get_output_activation(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)     
    
    # Move everything to GPU, if available
    if torch.cuda.is_available():
        move_to_gpu(problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
    # Get data [day,stock,feature]
    X_train, Y_train, Y_train_aux = problem.get_train_data()    # [200,50,28], [200,50], [200,50,50]
    X_val, Y_val, Y_val_aux = problem.get_val_data()            # [200,50,28], [200,50], [200,50,50]
    # val_rand = random.sample(range(len(X_val)), 2)
    # X_val, Y_val, Y_val_aux = X_val[val_rand], Y_val[val_rand], Y_val_aux[val_rand]
    X_test, Y_test, Y_test_aux = problem.get_test_data()        # [400,50,28], [400,50], [400,50,50]

    print('Ours Training Method...')
    if args.minmax.upper()=="MIN":
        best = (float("inf"), None)
    elif args.minmax.upper()=="MAX":
        best = (float("-inf"), None)
    
    else:
        raise LookupError()

    for epoch in tqdm(range(10000), desc='{}_training'.format(args.market)):  ##default 10000
        if epoch % args.valfreq == 0:
            # Check if well trained by objective value
            # datasets = [(X_train, Y_train, Y_train_aux, 'train'), (X_val, Y_val, Y_val_aux, 'val')]
            val_datasets = [(X_val, Y_val, Y_val_aux, 'val')]
            # metrics = metrics2wandb(datasets, model, problem, f"Iter {epoch}")
            metrics = print_metrics(val_datasets, model, problem, args.loss, loss_fn, f"Iter {epoch},", args.wandb)           
            # Save model if it's the best one
            if args.minmax.upper()=="MIN":
                if best[1] is None or metrics['val']['objective'] < best[0]:
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0
            else:
                if best[1] is None or metrics['val']['objective'] > best[0]:
                    # print(epoch)
                    best = (metrics['val']['objective'], deepcopy(model))
                    steps_since_best = 0

            # Stop if model hasn't improved for patience steps
            if (args.earlystopping) and (steps_since_best > args.patience):
                break
        
        #################### TEST ####################
        # Learn
        losses = []
        for i in random.sample(range(len(X_train)), min(args.batchsize, len(X_train))):
            pred = model(X_train[i]).squeeze()
            losses.append(loss_fn(pred, Y_train[i], aux_data=Y_train_aux[i], partition='train', index=i))
        # loss = torch.stack(losses).nanmean()   
        losses = torch.stack(losses)
        loss = losses.nansum()/(len(losses)-losses.isnan().sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps_since_best += 1
        
        wandb.log({"loss": loss})
        if epoch % 20 == 0 and args.loss == 'dfl':
            torch.save(model.state_dict(), 'dense_model/{}/{}_{}_epoch_{}_cardinality_{}_dflalpha_{}_layer_{}_lr_{}_batch_{}.pt'.format(args.market,args.loss,args.exp_num,epoch, args.cardinality, args.dflalpha,args.layers, args.lr, args.batchsize))
        ###############################################       
    if args.earlystopping:
        print("Early Stopping... Saving the Model...")
        model = best[1]      



    ## Save Prediction model
    print('Save final prediction model: {}'.format(args.loss))
    torch.save(model.state_dict(), 'dense_model/{}/{}_{}__cardinality_{}_dflalpha_{}_layer_{}_lr_{}_batch_{}_Final.pt'.format(args.market,args.loss,args.exp_num,args.cardinality,args.dflalpha, args.layers, args.lr, args.batchsize))