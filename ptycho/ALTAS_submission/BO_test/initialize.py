import sys
import os
import numpy as np
from parfile_generator import parfile
import ptycho_thread_starter
import shutil

# 12/10/21, cz, check if the results folders exist, remove them to avoid errors if they exist.
# 12/10/21, cz, change the format of setup files to also include the parameters that are fixed, and read the values in class parfile.
# 12/10/21, cz, add SOBO/MOBO selection to parameter file.
# 12/22/21, cz add more parameters to control the grouping parameter, and determine whether to run from previous results.

# TODO: add another parameter in the setup.txt to determine performing multislice/mixed-state
# TODO: add running time to the setup file too.
# TODO: add parameters to control probe_search_start.
# TODO: add job name to setup file and modify corresponding part of the submission file for each thread. Won't be able to change the job name on bo_thread.sub, as that is the entry point of everything.
# TODO, low priority: randomize the seed for the random number generator when initialize parameters.
# TODO, low priority: add a variable for the path of the parameter files. Currently, it is hard coded as the same path of the .py script.
# TODO, low priority: add another paramter in the setup.txt to change the general job name. Now the bo_thread.sub is the start point of everything, we need another python script to run the bo_thread.sub, then another .sub to run that python script. Seems like lots of trouble.


# ALTAS cluster requires the latest pytorch to use the A100 GPUs, install latest pytorch using:
# pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

def main(setup_file: str):
    '''
    setup_file: str, path to the setup text file which contains all the fixed and tunable parameters.
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
            val = par_dict[par][0] + par_dict[par][1] * (np.random.rand() - 0.5) * 2
            file.modify_parameter(par, str(val)) 
            count += 1
        
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
        
        parfile_name = 'parameter_thread' + str(i) + '_next.txt'
        file.save_file(parfile_name, '')

    # Launch reconstruction threads
    for i in range(njobs):
        ptycho_thread_starter.main(i)
    return

if __name__ == "__main__":
    main(sys.argv[1])