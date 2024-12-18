from PThenO import PThenO
from time import time
import pickle
from tqdm import tqdm
# import quandl
import datetime as dt
import pandas as pd
import os
import torch
import random
import pdb
import cvxpy as cp
import numpy as np
import itertools
from cvxpylayers.torch import CvxpyLayer
from cvxpy.reductions.solvers.solving_chain import construct_solving_chain
from data_process import load_data, load_target_index, load_bench_target_index, load_bench_data, load_prev_marketcap



class IndexTracking(PThenO):
    """A class that implements the Portfolio Optimization problem from
    Wang, Kai, et al. "Automatically learning compact quality-aware surrogates
    for optimization problems." Advances in Neural Information Processing
    Systems 33 (2020): 9586-9596.
    
    The code is largely adapted from: https://github.com/guaguakai/surrogate-optimization-learning/"""

    def __init__(
        self,
        num_train_instances=200,  # number of *days* to use from the dataset to train
        num_test_instances=200,  # number of *days* to use from the dataset to test
        num_stocks=50,  # number of stocks per instance to choose from
        val_frac=0.2,  # fraction of training data reserved for test
        rand_seed=0,  # for reproducibility
        K=10,  # cardinality constraint
        data_dir="data",  # directory to store data
        market ='nasdaq',
        save_mode = True
        
    ):
        super(IndexTracking, self).__init__()
        
        # Do some random seed fu
        self.rand_seed = rand_seed
        self._set_seed(self.rand_seed)

        # Load train and test labels
        self.num_stocks = num_stocks
        self.market = market
        self.Xs, self.Ys, self.marketcap_mat, self.future_mat = self._load_instances(data_dir, num_stocks, market)

        # Split data into train/val/test
        #   Sanity check and initialisations
        total_days = self.Xs.shape[0]
        self.num_train_instances = num_train_instances
        self.num_test_instances = num_test_instances
        num_days = self.num_train_instances + self.num_test_instances
        assert self.num_train_instances + self.num_test_instances < total_days
        assert 0 < val_frac < 1
        self.val_frac = val_frac
        
        

        #   Creating "days" for train/valid/test
        idxs = list(range(num_days))
        num_val = int(self.val_frac * self.num_train_instances)
        self.train_idxs = idxs[:self.num_train_instances - num_val]
        self.val_idxs = idxs[self.num_train_instances - num_val:self.num_train_instances]
        self.test_idxs = idxs[self.num_train_instances:]
        assert all(x is not None for x in [self.train_idxs, self.val_idxs, self.test_idxs])
        self.K = K
        # self.opt = self._create_cvxpy_problem(K=self.K)  ## We need input aux_data for each opt prob

        if self.num_stocks > self.Xs.shape[1]:
            self.num_stocks = self.Xs.shape[1]

        self.save_mode = save_mode
        if self.save_mode:
            self.prob_dict = self.load_cvxpy_problem(self.K, self.marketcap_mat[:,:self.num_stocks],self.train_idxs, self.val_idxs, self.test_idxs)
        
        
            
        
        # Undo random seed setting
        self._set_seed()

    def _load_instances(
        self,
        data_dir,
        stocks_per_instance,
        market,
        reg=0.1,
    ):
        # Get raw data
        feature_mat, target_mat, _, future_mat, _, dates, symbols = load_data(market=market)
        # idx_target_mat = load_target_index() #Load Target index SP500
        marketcap_mat = load_prev_marketcap(dates, market) #Load Asset market cap (for weight calculation)


        target_mat = target_mat.squeeze()
        # idx_target_mat = idx_target_mat.squeeze()

        # Normalize features
        num_features = feature_mat.shape[-1]
        feature_mat_flat = feature_mat.reshape(-1, num_features)
        feature_mat = torch.div((feature_mat - torch.mean(feature_mat_flat, dim=0)), (torch.std(feature_mat_flat, dim=0) + 1e-12))

        return feature_mat.float(), target_mat.float(), marketcap_mat.float(), future_mat.float()
    
    def _get_price_feature_df(
        self,
        overwrite=False,
    ):
        """
        Loads raw historical price data if it exists, otherwise compute the file on the fly, this adds other timeseries
        features based on rolling windows of the price
        :return:
        """
        print("Loading dataset...")
        price_feature_df = pd.read_csv(self.price_feature_file, index_col=["Date", "Symbol"])


        return price_feature_df
    
    
    def _get_price_feature_matrix(self, price_feature_df):
        num_dates, num_assets = map(len, price_feature_df.index.levels)
        price_matrix = price_feature_df.values.reshape((num_dates, num_assets, -1))
        return price_matrix


    def _create_cvxpy_problem(
        self,
        K,
        prev_market_cap
    ):
        
        W = cp.Variable((self.num_stocks,self.num_stocks), PSD = True) ## set psd constraint here
        r_hat_square = cp.Parameter((self.num_stocks, self.num_stocks)) #future_stock_price (prediction)
        constraints = [cp.sum(W) == 1, cp.sum(cp.abs(W)) <= K*cp.trace(W), W >> 0]
        objective = cp.Minimize((cp.trace(r_hat_square@ W)-2*cp.sum(W@r_hat_square@prev_market_cap)))  # Future target index I = r_hat * prev_marketcap
        problem = cp.Problem(objective, constraints)
        
        # Z (optimal solution(Variable)) for given parameter(Predicted Value)
        # return overall problem (including variables, parameters )
        return CvxpyLayer(problem, parameters=[r_hat_square], variables=[W])

    def load_cvxpy_problem(self, K, prev_market_cap,train_idxs, val_idxs, test_idxs):
        
        try:
            with open('problems/{}/Indextracking_{}_train.pkl'.format(self.market,K), 'rb') as f:
                prob_train = pickle.load(f)
            with open('problems/{}/Indextracking_{}_val.pkl'.format(self.market,K), 'rb') as f:
                prob_val = pickle.load(f)
            # with open('problems/{}/Indextracking_{}_test.pkl'.format(self.market,K), 'rb') as f:
            #     prob_test = pickle.load(f)
            
            # prob_dict = {'train': prob_train, 'val': prob_val, 'test': prob_test}
        
            prob_dict = {'train': prob_train, 'val': prob_val}
        
        except:
            prob_dict = {'train':[], 'val':[]}
            
            for idx in tqdm(train_idxs, desc = "train problem saving>>>"):
                cvxpy_prob = self._create_cvxpy_problem(K, prev_market_cap[idx])
                prob_lst = prob_dict['train']
                prob_lst.append(cvxpy_prob)
                
            with open('problems/{}/Indextracking_{}_train.pkl'.format(self.market,K), 'wb') as f:
                pickle.dump(prob_lst, f)
                
            for idx in tqdm(val_idxs, desc = "val problem saving>>>"):
                cvxpy_prob = self._create_cvxpy_problem(K, prev_market_cap[idx])
                prob_lst = prob_dict['val']
                prob_lst.append(cvxpy_prob)
                
            with open('problems/{}/Indextracking_{}_val.pkl'.format(self.market,K), 'wb') as f:
                pickle.dump(prob_lst, f)
            
            # for idx in tqdm(test_idxs, desc = "test problem saving>>>"):
            #     cvxpy_prob = self._create_cvxpy_problem(K, prev_market_cap[idx])
            #     prob_lst = prob_dict['test']
            #     prob_lst.append(cvxpy_prob)
                
            # with open('problems/{}/Indextracking_{}_test.pkl'.format(self.market,K), 'wb') as f:
            #     pickle.dump(prob_lst, f)
            
            
        return prob_dict
        

    def ret_2_ret_sq(self, ret):
        if ret.ndim == 1:
            ret_sq_mat = np.outer(ret,ret)
        else:
            for i in range(ret.shape[0]):
                if i == 0:
                    ret_sq_mat = np.expand_dims(np.outer(ret[i], ret[i]), axis=0)
                else:
                    ret_sq = np.expand_dims(np.outer(ret[i], ret[i]), axis=0)
                    ret_sq_mat = np.vstack((ret_sq_mat, ret_sq))
        
        return ret_sq_mat

    def get_train_data(self, **kwargs):
        return self.Xs[self.train_idxs,:self.num_stocks,:], self.Ys[self.train_idxs,:self.num_stocks], self.marketcap_mat[self.train_idxs,:self.num_stocks]

    def get_val_data(self, **kwargs):
        return self.Xs[self.val_idxs,:self.num_stocks,:], self.Ys[self.val_idxs,:self.num_stocks], self.marketcap_mat[self.val_idxs,:self.num_stocks]

    def get_test_data(self, **kwargs):
        return self.Xs[self.test_idxs,:self.num_stocks,:], self.Ys[self.test_idxs,:self.num_stocks], self.marketcap_mat[self.test_idxs,:self.num_stocks]
    
    def get_future_data(self, test_data_num,**kwargs):
        return self.future_mat[self.test_idxs[test_data_num], :self.num_stocks, :], self.marketcap_mat[self.test_idxs[test_data_num]+1, :self.num_stocks].T
    
    def get_bench_test_data(self, **kwargs):
        self.bench_idx_target_mat =  load_bench_target_index()
        self.bench_Ys = load_bench_data()
        return self.Xs[self.test_idxs,:self.num_stocks,:], self.bench_Ys[self.test_idxs,:self.num_stocks], self.bench_idx_target_mat[self.test_idxs,:self.num_stocks]
    
    def get_future_idx_test_data(self, **kwargs):
        self.future_idx_target_mat =  load_target_index()
        return self.future_idx_target_mat[self.test_idxs]

    def get_modelio_shape(self):
        return self.Xs.shape[-1], 1

    def get_twostageloss(self):
        return 'mse'

    def _get_covar_mat(self, instance_idxs):
        return self.idx_target_mat.reshape((-1, *self.idx_target_mat.shape[2:]))[instance_idxs]

    def get_decision(self, Y, aux_data, data_partition= 'a', data_num=100000 ,max_instances_per_batch=1500, **kwargs):
        try:
            if Y.get_device() != -1:
                Y = Y.detach().cpu()
        except:
            pass
        
        try:
            if aux_data.get_device() != -1:
                aux_data = aux_data.detach().cpu()
        except:
            pass
        
        if self.save_mode:
            problems = self.prob_dict[data_partition]
        else:
            pass
        Y_sq = torch.tensor(self.ret_2_ret_sq(np.array(Y)))

        assert Y_sq.ndim <= 3
        if Y_sq.ndim == 3:
            results = []
            for idx in tqdm(range(Y_sq.shape[0]), desc = '{}_validation process: '.format(self.market)):  
                if self.save_mode:          
                    problem = problems[idx]
                else:
                    problem = self._create_cvxpy_problem(self.K, aux_data[idx])
                
                W = problem(Y_sq[idx], solver_args = {'max_iters':int(5e3),'eps_abs':1e-5,'eps_rel':1e-5})[0]
                W = np.round(W,4) 
                w = torch.sqrt(torch.diagonal(W))
                rebalance_w = w/(w.sum())
                results.append(rebalance_w)
            return torch.stack(results)               
        
        else:
            if self.save_mode:
                problem = problems[data_num]
            else:
                problem = self._create_cvxpy_problem(self.K, aux_data)
            W = problem(Y_sq, solver_args = {'max_iters':int(5e3),'eps_abs':1e-5,'eps_rel':1e-5})[0]
            W = np.round(W,4)
            w = torch.sqrt(torch.diagonal(W))
            rebalance_w = w/(w.sum())
            return rebalance_w

    def get_objective(self, Y, Z, aux_data, train_test = 'train',**kwargs):
        try:
            if Y.get_device() != -1:
                Y = Y.detach().cpu()
        except:
            pass
        
        try:
            if aux_data.get_device() != -1:
                aux_data = aux_data.detach().cpu()
        except:
            pass 
        
        ## For Train, Inference : aux_data is "Previous Marketcap weight"
        if train_test == 'train':
            if Y.ndim ==2:
                Y_sq = torch.tensor(self.ret_2_ret_sq(np.array(Y)))
                W = torch.tensor(self.ret_2_ret_sq(np.array(Z)))
                objs = []
                for i in range(Y_sq.shape[0]):
                    obj = torch.trace(Y_sq[i]@W[i])-2*torch.sum(W[i]@Y_sq[i]@aux_data[i])
                    objs.append(obj)
                objs = torch.tensor(objs)
            else:
                Y_sq = torch.tensor(self.ret_2_ret_sq(np.array(Y)))
                W = torch.tensor(self.ret_2_ret_sq(np.array(Z)))
                objs = torch.trace(Y_sq@W)-2*torch.sum(W@Y_sq@aux_data)
            
            return objs
        ## For Test: aux_data is "Target index"
        else: 
            idx_target_mat = aux_data
            obj = (idx_target_mat- (Y * Z).sum(dim=-1)).square()
            return obj
    
    def get_output_activation(self):
        return 'tanh'


if __name__ == "__main__":
    problem = IndexTracking()
    X_train, Y_train, Y_train_aux = problem.get_train_data()

    Z_train = problem.get_decision(Y_train, aux_data=Y_train_aux)
    obj = problem.get_objective(Y_train, Z_train, aux_data=Y_train_aux)

    pdb.set_trace()