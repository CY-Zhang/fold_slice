import sys
import os
import numpy as np
import scipy.io as sio
from parfile_generator import parfile

# TODO: also prepare the slurm script using this python script, then run the .sub file with initialized files.
# Currently, the PARFILELIST in the slurm script needs to be manually changed to match the parameter files created here.
def main(setup_file: str):
    '''
    data_file: string, .mat file with the 4D array name as 'cbed'.
    npar: integer, number of parameter jobs to be submitted.
    '''
    with open(setup_file) as f:
        lines = f.readlines()
        f.close()
    data_file = lines[0].split(' ')[1][:-1]
    result_path = lines[1].split(' ')[1][:-1]
    njobs = int(lines[2].split(' ')[1][:-1])
    par_dict = {}
    for i in range(3, len(lines)):
        par = lines[i].split(' ')
        par_dict[par[0]] = [float(par[1]), float(par[2][:-1])]

    # split the file into two parts and save separately, name as ..._0.mat and ..._1.mat.
    data_splitter(data_file)

    # initialize the train_x file
    train_x = np.zeros([njobs//2, len(lines) - 3])
    train_x_next = train_x

    # setup two sets of initial parameters for initial run
    njobs = njobs // 2
    filename1 = data_file.split('.')[0] + '_0.mat'
    filename2 = data_file.split('.')[0] + '_1.mat'
    for i in range(njobs):
        file1 = parfile(1, filename1, result_path + str(i) + '_0/') # initialize for each loop with different saving path
        file2 = parfile(1, filename2, result_path +str(i) + '_1/')
        os.mkdir(result_path + str(i) + '_0/')
        os.mkdir(result_path + str(i) + '_1/')
        # create pairs of parameter files with random parameters
        count = 0
        for par in par_dict:
            val = par_dict[par][0] + par_dict[par][1] * (np.random.rand() - 0.5) * 2
            train_x[i, count] = val
            # TODO: randomize the seed for the random number
            file1.modify_parameter(par, val) 
            file2.modify_parameter(par, val)
            # TODO: change to auto number instead of hard coded 30.
            file1.modify_parameter('N_scan_y', 30)
            file2.modify_parameter('N_scan_y', 30)
            count += 1
        # TODO: add a variable for the path of the parameter files. Currently, it is hard coded as the same path of the .py script.
        parfile_name_1 = 'parameter_MoS2_' + str(i+1) + '_0.txt'
        parfile_name_2 = 'parameter_MoS2_' + str(i+1) + '_1.txt'
        file1.save_file(parfile_name_1, '')
        file2.save_file(parfile_name_2, '')
    # save the array of train_x
    np.save(result_path + 'par_X.npy', train_x)

    # setup two sets of initial parameters for the next run
    for i in range(njobs):
        file1 = parfile(1, filename1, result_path + str(i) + '_0/')
        file2 = parfile(1, filename2, result_path + str(i) + '_1/')
        count = 0
        for par in par_dict:
            val = par_dict[par][0] + par_dict[par][1] * (np.random.rand() - 0.5) * 2
            train_x_next[i, count] = val
            file1.modify_parameter(par, val) 
            file2.modify_parameter(par, val)
            count += 1
        parfile_name_1 = 'parameter_MoS2_' + str(i+1) + '_0_next.txt'
        parfile_name_2 = 'parameter_MoS2_' + str(i+1) + '_1_next.txt'
        file1.save_file(parfile_name_1, '')
        file2.save_file(parfile_name_2, '')
    # save the array of train_x_next
    np.save(result_path + 'par_X_next.npy', train_x_next)

    return

def data_splitter(data_file):
    data = sio.loadmat(data_file)
    data1 = data[:,:,:,0::2]
    data2 = data[:,:,:,1::2]
    filename1 = data_file.split('.')[0] + '_0.mat'
    filename2 = data_file.split('.')[0] + '_1.mat'
    mdic = {"cbed": data1}
    sio.savemat(filename1, mdic)
    mdic = {"cbed": data2}
    sio.savemat(filename2, mdic)
    return

if __name__ == "__main__":
    main(sys.argv[1])