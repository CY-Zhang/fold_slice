% This script prepares experimental electron ptycho. data for PtychoShelves
function prepare_data(parfile)
    %% Step 1: download sample data (rawdata_1x_crop.mat) from the link provided in 
    % https://www.nature.com/articles/s41467-020-16688-6
    % Note: it's a good practice to store data (and reconstructions) in a
    % different folder from fold_slice 
    
    %% Step 2: load data
    par = parameter_builder(parfile);  % load the struct with parameters
    load(par.raw_data);
    
    %% Step 3: go back to .../fold_slice/ptycho and pre-process data
    % load the parameters in case it is not saved in the raw data
    df = par.defocus;
    voltage = par.voltage;
    rbf = par.rbf;
    ADU = par.ADU;
    alpha0 = par.alpha_max;
    
    addpath(strcat(pwd,'/utils_electron/'))
    Np_p = [par.CBED_size, par.CBED_size]; % size of diffraction patterns used during reconstruction. can also pad to 256
    % pad cbed
    [ndpy,ndpx,npy,npx]=size(cbed);
    if ndpy < Np_p(1) % pad zeros
        dp=padarray(cbed,[(Np_p(1)-ndpy)/2,(Np_p(2)-ndpx)/2,0,0],0,'both');
    else
        dp=crop_pad(cbed,Np_p);
    end
    
    dp = dp / ADU; % convert to electron count
    dp=reshape(dp,Np_p(1),Np_p(2),[]);
    Itot=mean(squeeze(sum(sum(dp,1),2))); %need this for normalizting initial probe
    
    % calculate pxiel size (1/A) in diffraction plane
    [~,lambda]=electronwavelength(voltage);
    dk=alpha0/1e3/rbf/lambda; %%% PtychoShelves script needs this %%%
    
    %% Step 4: save CBED in a .hdf5 file (needed by Ptychoshelves)
    scan_number = par.scan_number; %Ptychoshelves needs
    save_dir = strcat(par.result_dir,num2str(scan_number),'/');
    disp(save_dir);
    mkdir(save_dir)
    roi_label = par.roi_label;
    saveName = strcat('data_roi',roi_label,'_dp.hdf5');
    if ~isfile(saveName)
        h5create(strcat(save_dir,saveName), '/dp', size(dp),'ChunkSize',[size(dp,1), size(dp,2), 1],'Deflate',4)
        h5write(strcat(save_dir,saveName), '/dp', dp)
    end
    
    %% Step 5: prepare initial probe
    dx=1/Np_p(1)/dk; %% pixel size in real space (angstrom)
    
    par_probe = {};
    par_probe.df = df;
    par_probe.voltage = voltage;
    par_probe.alpha_max = alpha0;
    par_probe.plotting = true;
    probe = make_tem_probe(dx, Np_p(1), par_probe);
    
    probe=probe/sqrt(sum(sum(abs(probe.^2))))*sqrt(Itot)/sqrt(Np_p(1)*Np_p(2));
    probe=single(probe);
    % add parameters for PtychoShelves
    p = {};
    p.binning = false;
    p.detector.binning = false;
    
    %% Step 6: save initial probe
    save(strcat(save_dir,'/init_probe.mat'),'probe','p')
end