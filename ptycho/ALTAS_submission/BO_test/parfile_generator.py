class parfile():
    def __init__(self, savepath, setup_file) -> None:
        '''
        Initialize a parfile class to edit and save parameter files.
        Inputs:
        option: 1 for mixed-state, 2 for multislice.
        datafile: full path to the .mat rawdata.
        savepath: folder to save the results.
        '''
        self.par_dict = {
        'result_dir': savepath,
        }
        with open(setup_file) as f:
            lines = f.readlines()
            f.close()
        data_file = lines[0].split(' ')[1][:-1]
        self.par_dict['raw_data'] = data_file
        for i in range(4, len(lines)):
            par = lines[i].split(' ')
            self.par_dict[par[0]] = par[1][:-1]

    def save_file(self, path: str, description: str) -> None:
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

    def modify_parameter(self, par_name: str, value: str) -> None:
        '''
        Function that modifies a single parameter.
        '''
        if par_name not in self.par_dict:
            return
        self.par_dict[par_name] = value
        return

    def load_file(self, path) -> None:
        pass