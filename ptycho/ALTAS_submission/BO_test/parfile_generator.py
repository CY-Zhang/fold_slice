class parfile():
    def __init__(self, option, datafile, savepath) -> None:
        '''
        Initialize a parfile class to edit and save parameter files.
        Inputs:
        option: 1 for mixed-state, 2 for multislice.
        datafile: full path to the .mat rawdata.
        savepath: folder to save the results.
        '''
        self.option = option
        self.par_dict = {'raw_data': datafile,
        'result_dir': savepath,
        'roi_label': '0_Ndp128',
        'voltage': '80',
        'alpha_max': '21.4',
        'rbf': '26',
        'defocus': '-500',
        'Nprobe': '5',
        'N_scan_x': '60',
        'N_scan_y': '60',
        'scan_step_size': '0.85',
        'Niter': '500',
        'Niter_save_results': '200',
        'ADU': '151',
        'rot_ang': '30',
        'CBED_size': '128',
        'extra_print_info': 'MoS2',
        'scan_number': '1',
        'gpu_id': '1',
        'thickness' : '210',
        'Nlayer' : '21'}

    def save_file(self, path, description) -> None:
        '''
        Function that save a new parameter file to the desired path.
        Input:
        path: string, full path of the new parameter file to be saved, suffix included.
        description: string, description line of the parameter file.
        '''
        # print(self.par_dict)
        with open(path, 'w') as f:
            f.write(description)
            f.write('\n')
            for item in self.par_dict:
                output_str = str(item) + ' '+ str(self.par_dict[item])
                f.write(output_str)
                f.write('\n')
        f.close()
        return

    def modify_parameter(self, par_name, value) -> None:
        '''
        Function that modifies a single parameter.
        '''
        if par_name not in self.par_dict:
            return
        self.par_dict[par_name] = value
        return

    def load_file(self, path) -> None:
        pass