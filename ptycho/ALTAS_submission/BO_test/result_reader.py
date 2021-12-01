import sys
import os
import numpy as np
import scipy.io as sio

def main(setup_file: str):
    with open(setup_file) as f:
        lines = f.readlines()
        f.close()
    result_path = lines[1].split(' ')[1][:-1]
    njobs = int(lines[2].split(' ')[1][:-1])
    njobs = njobs//2

    # concatenate par_x into train_x, then replace par_x with par_x_next
    par_x = np.load(result_path + 'par_X.npy')
    if os.file.exists(result_path + 'train_X.npy'):
        train_x = np.load(result_path + 'train_X.npy')
        # TODO: could get error if one of the x matrices has only 1 row.
        train_x = np.concatenate((train_x, par_x), axis = 0)
    else:
        train_x = par_x
    np.save(result_path + 'train_X.npy', train_x)
    os.replace(result_path + 'par_X_next.npy', result_path + 'par_X.npy')

    # read new train_y data from the reconstruction results.
    # append new results into the train_y file, if train_y file does not exist, create a new one.

    # replace parameter files with next batch of parameter files.
    for i in range(njobs):
        parfile = 'parameter_MoS2_' + str(i+1) + '_0.txt'
        newfile = 'parameter_MoS2_' + str(i+1) + '_0_next.txt'
        os.replace(newfile, parfile)
        parfile = 'parameter_MoS2_' + str(i+1) + '_1.txt'
        newfile = 'parameter_MoS2_' + str(i+1) + '_1_next.txt'
        os.replace(newfile, parfile)
    return

def calculate_fsc(image1, image2):
    return

if __name__ == "__main__":
    main(sys.argv[1])