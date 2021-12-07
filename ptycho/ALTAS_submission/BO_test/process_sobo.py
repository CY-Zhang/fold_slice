from botorch.sampling.samplers import SobolQMCNormalSampler
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.utils.transforms import unnormalize

import time
import numpy as np
import os
from parfile_generator import parfile
import sys

#TODO: figure out a better way to create multiple acuqisition points, the current way often create very close points.
def main(setup_file: str, round: int):

    # try to read the setup file
    while True:
        try:
            f = open(setup_file)
            break
        except IOError:
            time.sleep(1)

    lines = f.readlines()
    f.close()
    data_file = lines[0].split(' ')[1][:-1]
    result_path = lines[1].split(' ')[1][:-1]
    n_thread = int(lines[2].split(' ')[1][:-1]) - 1
    par_dict = {}
    for i in range(3, len(lines)):
        par = lines[i].split(' ')
        par_dict[par[0]] = [float(par[1]), float(par[2][:-1])]
    print("Setting file loaded by BO thread.\n")

    # check whether the current batch is finished
    while True:
        if check_ready(n_thread):
            break
        time.sleep(10)
    print("Start optimization round "+ str(round) + '.\n')

    # when the batch is finished, read in train_X and train_Y.
    while True:
        try:
            train_x = np.load(result_path + 'train_X.npy')
            break
        except IOError:
            time.sleep(1)
    train_y = np.load(result_path + 'train_Y.npy')
    bounds = [[],[]]
    for i in par_dict:
        bounds[0].append(par_dict[i][0] - par_dict[i][1])
        bounds[1].append(par_dict[i][0] + par_dict[i][1])
    new_x = predict_next(train_x, train_y, n_thread, bounds)

    # save new jobs from the new_x predicted by BO.
    for i in range(n_thread):
        file = parfile(1, data_file, result_path + 'thread_' + str(i) + '/') # initialize for each loop with different saving path
        # create pairs of parameter files with random parameters
        idx = 0
        for par in par_dict:
            val = new_x[i][idx]
            file.modify_parameter(par, val) 
            idx += 1
        # TODO: add a variable for the path of the parameter files. Currently, it is hard coded as the same path of the .py script.
        parfile_name = 'parameter_thread' + str(i) + '_next.txt'
        file.save_file(parfile_name, '')
    return

def predict_next(train_x, train_y, n_predict, bounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.tensor(train_x[1:], device = device)
    train_y = torch.tensor(train_y[1:], device = device)
    bounds = torch.tensor(bounds, device = device)
    print(train_x.shape)

    # normalize train_x and train_y
    train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
    # train_y = (train_y - min(train_y)) / (max(train_y) - min(train_y)) if len(train_y) > 1 else train_y / max(train_y)

    if len(train_y.shape) == 1: # case of single objective
        train_y = torch.tensor(train_y).unsqueeze(-1)

    # gp = SingleTaskGP(train_x, train_y)
    outcome_transformer = Standardize( m = 1,
    batch_shape = torch.Size([]),
    min_stdv = 1e-08)
    # TODO: figure out how to make transformer work with multiple candidate outputs
    gp = SingleTaskGP(train_x, train_y, outcome_transform = outcome_transformer)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    # TODO: change beta of UCB into a variable, and 2 might be too much for parameter tuning.
    UCB = UpperConfidenceBound(gp, beta = 2)
    # sampler = SobolQMCNormalSampler(1024)
    # qUCB = qUpperConfidenceBound(gp, 2, sampler)
    candidate, _ = optimize_acqf(
        UCB, bounds=torch.stack([torch.zeros(train_x.shape[1], device = device), torch.ones(train_x.shape[1], device = device)]), 
        q = 1, 
        num_restarts=n_predict, raw_samples=20, 
    )
    new_x =  unnormalize(candidate.detach(), bounds=bounds)
    print(new_x)
    return new_x.squeeze().cpu().detach().numpy()

def check_ready(n_thread: int):
    for i in range(n_thread):
        filename = 'parameter_thread' + str(i) + '_next.txt'
        if os.path.exists(filename):
            return False
    return True

if __name__ == "__main__":
    # main(sys.argv[1], sys.argv[2])
    main('setup.txt', 0) # for debug use