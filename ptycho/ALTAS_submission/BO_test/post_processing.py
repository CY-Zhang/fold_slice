import os
import sys
import numpy as np
import scipy.io as sio
import time
from PIL import Image
import shutil

# 12/21/21 added paramter option_mobo to determine how many objectives to return.
# 12/21/21, cz, automatically determines the path to read the tif file, current scheme only works when there is only one folder under each scan number. Also auto determine the niter part of filename from setup file.
# 12/23/21, cz, modify the roi_label and scan number part, read them from the setup file instead of hard coded as roi0_Ndp128 and 1.

# TODO: ignore lines in the setup files that starts with #
# TODO: keep the hdf5 files, remove init_probe and the whole folder named roi_label, could take long to generate the two files again for massive data size.
def main(setup_file: str, thread_idx: int):

    old_file = 'parameter_thread' + str(thread_idx) + '.txt'

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
    option_mobo = True if method == 'mobo' else False
    par_dict = {}
    n_iter = 500
    for i in range(3, len(lines)):
        par = lines[i].split(' ')
        if par[0] == 'Niter':
            n_iter = int(par[1][:-1])
        if par[0] == 'roi_label':
            roi_label = par[1][:-1]
        if par[0] == 'scan_number':
            scan_number = par[1][:-1]
        if len(par) == 3:
            par_dict[par[0]] = 0

    # try to open and edit the train_X and trin_Y file
    while True:
        try:
            train_x = np.load(result_path + 'train_X.npy')
            train_y = np.load(result_path + 'train_Y.npy')
            break
        except IOError:
            time.sleep(1)

    # check if the output file exist, if not, the reconstruction failed, remove the files, modify the input, then go back and run again. Keep next parameter in place, so that the BO process won't start.
    thread_path = result_path + 'thread_' + str(thread_idx) + '/' + scan_number + '/roi' + roi_label + '/'
    image_path = thread_path + next(os.walk(thread_path))[1][0]
    image_file = image_path + '/obj_phase_roi/obj_phase_roi_Niter' + str(n_iter) + '.tiff'
    if not os.path.exists(image_file):
        image_file = image_path + '/obj_phase_roi_sum/obj_phase_roi_sum_Niter' + str(n_iter) + '.tiff'
    if not os.path.exists(image_path):
    # call BO to give a new parameter, if train_x is empty, create a random one.
        pass

    # compute the new train_X and train_Y
    par_dict = get_train_X(par_dict, old_file)
    new_x = np.array([float(par_dict[i]) for i in par_dict])
    angle = get_conv_angle(old_file)
    new_y = get_train_Y(thread_path, angle, n_iter, option_mobo)

    # concatenate par_x into train_x, then replace par_x with par_x_next
    # TODO: could get error if one of the x matrices has only 1 row.
    train_x = np.concatenate((train_x, [new_x]), axis = 0)
    train_y = np.concatenate((train_y, [new_y]), axis = 0)
    np.save(result_path + 'train_X.npy', train_x)
    np.save(result_path + 'train_Y.npy', train_y)

    # copy the final phase of the object and save the parameters in the filename
    filename = ""
    if option_mobo:
        filename += 'error_'+ "{:.4f}".format(new_y[0])
        filename += '_FSC_' + "{:.4f}".format(new_y[1])
    else:
        filename += 'error_'+ "{:.4f}".format(new_y)
    for i in par_dict:
        filename += ('_' + i + '_' + "{:.3f}".format(float(par_dict[i])))
    filename += '.tiff'
    shutil.copyfile(image_file, result_path + filename)

    # Keep detecting whether the next batch of parameters are ready.
    newfile = 'parameter_thread' + str(thread_idx) + '_next.txt'
    while True:
        if os.path.exists(newfile):
            break
        time.sleep(1)

    # remove the result path after reading the x and y values
    shutil.rmtree(result_path + 'thread_' + str(thread_idx) + '/' + scan_number + '/')
    shutil.rmtree(result_path + 'thread_' + str(thread_idx) + '/analysis/')

    # replace parameter files with next batch of parameter files.
    os.replace(newfile, old_file)
    return

def get_conv_angle(par_file):
    with open(par_file) as f:
        lines = f.readlines()
        f.close()
    for line in lines[1:]:
        temp = line.split(' ')
        if temp[0] == 'alpha_max':
            return float(temp[1][:-1])
    return -1

def get_train_X(par_dict, par_file):
    with open(par_file) as f:
        lines = f.readlines()
        f.close()
    full_par_dict = {}
    for line in lines[1:]:
        temp = line.split(' ')
        par, val = temp[0], temp[1]
        full_par_dict[par] = val
    for key in par_dict:
        par_dict[key] = full_par_dict[key]
    return par_dict

def get_train_Y(path, angle, n_iter, option_mobo):
    recon_error = get_recon_error(path, n_iter)
    # TODO: calculate the wavelength from voltage.
    k_px = angle / 1000 / 26 / (4.18/100)
    r_px = 1 / k_px / 128
    image_path = path + next(os.walk(path))[1][0]
    image_file = image_path + '/obj_phase_roi/obj_phase_roi_Niter' + str(n_iter) + '.tiff'
    if not os.path.exists(image_file):
        image_file = image_path + '/obj_phase_roi_sum/obj_phase_roi_sum_Niter' + str(n_iter) + '.tiff'
    image = np.array(Image.open(image_file))
    if image.shape[0] % 2 != 0:
            image = image[:image.shape[0]//2*2,:]
    # 12-09-21, cz, new way to calculate FSC using the mean of abs coefficient without cropping image
    # Crop 10% edge part to remove random noise
    # height, width = image.shape
    # image = image[height//10:height//10*9, width//10:width//10*9]
    input1 = image[:-1, :-1]
    input2 = image[1:,1:]
    # for now, do not calibrate with pixel size, i.e. don't use d = r_px when calculating the frequency.
    # freq = np.fft.fftfreq(int(len(input1) * np.sqrt(2)), d = r_px)
    # idx = compute_frc_score(input1, input2, bin_width = 2, threshold = 0.143)
    frc, _ = compute_frc(input1, input2, bin_width = 2.0)
    if not option_mobo:
        return -np.log(recon_error)
    else:
        return np.array([ -np.log(recon_error), sum(np.abs(frc[~np.isnan(frc)]))/sum(~np.isnan(frc))])

def get_recon_error(path, n_iter):
    final_path = path + next(os.walk(path))[1][0]
    data = sio.loadmat(final_path + '/Niter' + str(n_iter) + '.mat' )
    error = data['outputs']['fourier_error_out']
    return error[0][0][-1][0]

def compute_frc_score(image_1, image_2, bin_width, threshold) -> int:
    frc, frc_bins = compute_frc(image_1, image_2, bin_width = bin_width)
    i = len(frc) - 1
    while i > 0:
        if np.isnan(frc[i]):
            i -= 1
            continue
        if frc[i] > threshold:
            break
        i -= 1
    
    return int(frc_bins[i])

def compute_frc(
        image_1: np.ndarray,
        image_2: np.ndarray,
        bin_width: int = 2.0
):
    """ Computes the Fourier Ring/Shell Correlation of two 2-D images

    :param image_1:
    :param image_2:
    :param bin_width:
    :return:
    """
    image_1 = image_1 / np.sum(image_1)
    image_2 = image_2 / np.sum(image_2)
    f1, f2 = np.fft.fft2(image_1), np.fft.fft2(image_2)
    af1f2 = np.real(f1 * np.conj(f2))
    af1_2, af2_2 = np.abs(f1)**2, np.abs(f2)**2
    nx, ny = af1f2.shape
    x = np.arange(-np.floor(nx / 2.0), np.ceil(nx / 2.0))
    y = np.arange(-np.floor(ny / 2.0), np.ceil(ny / 2.0))
    distances = list()
    wf1f2 = list()
    wf1 = list()
    wf2 = list()
    for xi, yi in np.array(np.meshgrid(x,y)).T.reshape(-1, 2):
        distances.append(np.sqrt(xi**2 + xi**2))
        xi = int(xi)
        yi = int(yi)
        wf1f2.append(af1f2[xi, yi])
        wf1.append(af1_2[xi, yi])
        wf2.append(af2_2[xi, yi])

    bins = np.arange(0, np.sqrt((nx//2)**2 + (ny//2)**2), bin_width)
    f1f2_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1f2
    )
    f12_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf1
    )
    f22_r, bin_edges = np.histogram(
        distances,
        bins=bins,
        weights=wf2
    )
    # TODO: f12_r * f22_r could result in zeros and thus nan in density.
    density = f1f2_r / np.sqrt(f12_r * f22_r)
    return density, bin_edges

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
    # main('setup.txt', 2)