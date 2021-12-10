import sys
import os
import numpy as np
import scipy.io as sio
from parfile_generator import parfile
import ptycho_thread_starter
import shutil

# TODO: check if the results folders exist, remove them to avoid errors if they exist.
# TODO: change the format of setup files to also include the parameters that are fixed, and read the values in class parfile.
# TODO: also prepare the slurm script using this python script, then run the .sub file with initialized files.
# TODO: add another paramter in the setup.txt to change the general job name. Now the bo_thread.sub is the start point of everything, we need another python script to run the bo_thread.sub, then another .sub to run that python script. Seems like lots of trouble.
# Done: add SOBO/MOBO selection to parameter file. Finished 12/10/21, cz.
# TODO: add another parameter in the setup.txt to determine performing multislice/mixed-state

# ALTAS cluster requires the latest pytorch to use the A100 GPUs, install latest pytorch using:
# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

def main(setup_file: str):
    '''
    data_file: string, .mat file with the 4D array name as 'cbed'.
    npar: integer, number of parameter jobs to be submitted.
    '''
    # read the job setups
    with open(setup_file) as f:
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

    # save empty train_x and train_y files
    train_x = np.zeros([1, count])
    train_y = np.expand_dims(np.zeros([2]), axis = 0) if option_mobo else np.zeros([1])
    np.save(result_path + 'train_X.npy', train_x)
    np.save(result_path + 'train_Y.npy', train_y)

    # setup initial parameter files
    file = parfile(result_path, setup_file)
    for i in range(njobs):
        thread_path = result_path + 'thread_' + str(i) + '/'
        if os.path.isdir(thread_path):
            shutil.rmtree(thread_path)
        os.mkdir(thread_path)
        file.par_dict['result_dir'] = thread_path
        # create pairs of parameter files with random parameters
        count = 0
        for par in par_dict:
            # TODO: randomize the seed for the random number
            val = par_dict[par][0] + par_dict[par][1] * (np.random.rand() - 0.5) * 2
            file.modify_parameter(par, str(val)) 
            count += 1
        # TODO: add a variable for the path of the parameter files. Currently, it is hard coded as the same path of the .py script.
        parfile_name = 'parameter_thread' + str(i) + '.txt'
        file.save_file(parfile_name, '')

    # setup next batch of parameter files
    for i in range(njobs):
        thread_path = result_path + 'thread_' + str(i) +'/'
        file.par_dict['result_dir'] = thread_path      
        # create pairs of parameter files with random parameters
        count = 0
        for par in par_dict:
            val = par_dict[par][0] + par_dict[par][1] * (np.random.rand() - 0.5) * 2
            file.modify_parameter(par, str(val))
            count += 1
        # TODO: add a variable for the path of the parameter files. Currently, it is hard coded as the same path of the .py script.
        parfile_name = 'parameter_thread' + str(i) + '_next.txt'
        file.save_file(parfile_name, '')

    # Launch reconstruction threads
    for i in range(njobs):
        ptycho_thread_starter.main(i)
    return

if __name__ == "__main__":
    main(sys.argv[1])