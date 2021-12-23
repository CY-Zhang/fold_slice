from botorch.sampling.samplers import SobolQMCNormalSampler
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.transforms import unnormalize

import time
import numpy as np
import os
from parfile_generator import parfile
import sys

# TODO: figure out a better way to create multiple acuqisition points, the current way often create very close points.
# TODO: move single-objective prediction to a separate function, and implement multi-batch.
# TODO: figure out how to make transformer work with multiple candidate outputs
# TODO: change beta of UCB into a variable, and 2 might be too much for parameter tuning.
# TODO: think about what to use as the reference point for MOBO.
# TODO, low priority: add a variable for the path of the parameter files. Currently, it is hard coded as the same path of the .py script.
# Python > 3.7.0 and botorch 0.5 are required for the multi_objective functions.
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
    result_path = lines[1].split(' ')[1][:-1]
    method = lines[2].split(' ')[1][:-1]
    njobs = int(lines[3].split(' ')[1][:-1]) - 1
    option_mobo = True if method == 'mobo' else False
    par_dict = {}
    count = 0  # counter to count the number of varying parameters.
    for i in range(4, len(lines)):
        par = lines[i].split(' ')
        if len(par) == 3:
            count += 1
            par_dict[par[0]] = [float(par[1]), float(par[2][:-1])]

    print("Setting file loaded by BO thread.\n")

    # check whether the current batch is finished
    while True:
        if check_ready(njobs):
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
    new_x = predict_next(train_x, train_y, njobs, bounds)

    # save new jobs from the new_x predicted by BO.
    file = parfile(result_path, setup_file)
    for i in range(njobs):
        thread_path = result_path + 'thread_' + str(i) + '/'
        file.par_dict['result_dir'] = thread_path
        # create pairs of parameter files with random parameters
        idx = 0
        for par in par_dict:
            val = new_x[i][idx]
            file.modify_parameter(par, val) 
            idx += 1
        parfile_name = 'parameter_thread' + str(i) + '_next.txt'
        file.save_file(parfile_name, '')
    return

def predict_next(train_x, train_y, n_predict, bounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.tensor(train_x[1:], device = device)
    train_y = torch.tensor(train_y[1:], device = device)
    bounds = torch.tensor(bounds, device = device)
    train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
    print(train_x.shape)

    if len(train_y.shape) == 1: # case of single objective
        
        train_y = torch.tensor(train_y).unsqueeze(-1)
        outcome_transformer = Standardize( m = 1,
        batch_shape = torch.Size([]),
        min_stdv = 1e-08)

        gp = SingleTaskGP(train_x, train_y, outcome_transform = outcome_transformer)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_model(mll)

        UCB = UpperConfidenceBound(gp, beta = 0.1)
        # sampler = SobolQMCNormalSampler(1024)
        # qUCB = qUpperConfidenceBound(gp, 2, sampler)
        candidate, _ = optimize_acqf(
            UCB, bounds=torch.stack([torch.zeros(train_x.shape[1], device = device), torch.ones(train_x.shape[1], device = device)]), 
            q = 1, num_restarts=n_predict, raw_samples=20, return_best_only=False,
        )
        new_x = candidate.detach()
        new_x = new_x * (bounds[1] - bounds[0]) + bounds[0]
    else:
        # MOBO case, train_x needs to be normalized to [0,1] before feed to the model.
        mll_qehvi, model_qehvi = initialize_model(train_x, train_y)
        fit_gpytorch_model(mll_qehvi)
        qehvi_sampler = SobolQMCNormalSampler(num_samples = 128)
        new_x = optimize_qehvi_and_get_observation(model_qehvi, train_y, qehvi_sampler, n_predict, bounds)

    print(new_x)
    return new_x.squeeze().cpu().detach().numpy()

def optimize_qehvi_and_get_observation(model, train_y, sampler, n_predict, bounds):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_RESTARTS = 20
    RAW_SAMPLES = 1024
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    ref_point = torch.tensor(np.zeros(train_y.shape[1]), device = device)
    partitioning = NondominatedPartitioning(ref_point=ref_point, Y=train_y)
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),  # use known reference point 
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([torch.zeros(bounds.shape[1], device = device), torch.ones(bounds.shape[1], device = device)]), 
        q = n_predict,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=bounds)
    return new_x

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def check_ready(n_thread: int):
    for i in range(n_thread):
        filename = 'parameter_thread' + str(i) + '_next.txt'
        if os.path.exists(filename):
            return False
    return True

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    # main('setup.txt', 0) # for debug use