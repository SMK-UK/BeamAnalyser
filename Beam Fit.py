'''
Sean Keenan, 5th Year MPhys Heriot-Watt University, Edinburgh
Mazerra group
Gaussian Beam Profile Extraction
'''

# import relevant modules
import os
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as mp
from scipy.optimize import curve_fit

# define neccesary functions
def gaussian(x, height, centre, sigma):
    '''Generates Gaussian with given parameters'''
    return height * np.exp(-(np.power(x - centre, 2) / (2 * sigma ** 2)))

# fit a gaussian to data by calculating its 'moments' (mean, variance, width, height)
def moments(data):
    '''Calculates parameters of a gaussian function by calculating
    its moments (height, mean_x, width_x, mean_y, width_y'''
    height = np.amax(data)
    centre = np.where(data == height)
    dim = np.size(centre)
    # handle case where there are multiple 'max' values
    if dim > 2:
        centre = np.array(centre)
        mean = np.sum(centre, 1)
        mean_x = round(mean[0] / len(centre[0]))
        mean_y = round(mean[1] / len(centre[1]))
    else:
        mean_x = int(centre[0])
        mean_y = int(centre[1])

    row = data[:, mean_y]
    col = data[mean_x, :]
    sigma_x = np.sqrt(((row - height) ** 2).sum() / len(row))
    sigma_y = np.sqrt(((col - height) ** 2).sum() / len(col))
    return height, mean_x, sigma_x, mean_y, sigma_y

def fitgauss(data):
    '''Returns seperate x-y Gaussian parameters from fit to 2D gaussian data
     (height, mean_x, width_x, mean_y, width_y)'''
    params = moments(data)
    # extract data along index of maximum value
    x = data[:, params[3]]
    y = data[params[1], :]
    # fit gaussian to data and return probability
    fit_x, success_x = curve_fit(gaussian, np.arange(1, len(x) +1 ), x, p0=params[0:3])
    fit_y, success_y = curve_fit(gaussian, np.arange(1, len(y) +1 ), y, p0=(params[0], params[3], params[4]))
    x_err = np.sqrt(np.diag(success_x))
    y_err = np.sqrt(np.diag(success_y))
    # condense fit data into array for output
    fit_data = np.array([fit_x, fit_y])
    fit_err = np.array([x_err, y_err])
    return fit_data, fit_err

''' Set-up Image file names and processing information '''

# directory name for images
path = '/Users/Message/Desktop/CCD data/20cm lens/'
file_list = os.listdir(path)
# extract relevant files and sort
image_list = natsorted([i for i in file_list if i.endswith('.bmp')])
xl_file = [i for i in file_list if i.endswith('.xls')]
# determine array dimensions for new data
img = Image.open(path + image_list[0])
imsize = img.size
# read image position (requires xlrd)
xlsx = pd.ExcelFile(path + xl_file[0])
z_pos = pd.read_excel(xlsx, sheet_name='Sheet1', header=2, usecols=['Distance (mm)'])
z = np.unique(z_pos.to_numpy())
# chip and pixel size (mm)
chip_size = 1/3 * 25.4
pix_size = 0.0045 # chip_size / imsize[0]
# image laser wavelength
wavelen = 606e-9

''' Read image and subtract background data - wont work if uneven number of files in folder '''

# read image and subtract background - store in new array
data = np.empty([int(len(image_list)/2), imsize[0], imsize[1]])
fit_data = np.empty([int(len(image_list)/2), 2, 3])
fit_err = np.empty([int(len(image_list)/2), 2, 3])
for index, image in enumerate(image_list):
    # break when half way through data
    if index < int(len(image_list)/2):
        # read then discard image (change to int32 as uint8 does not have -ve values)
        img = np.int32(np.transpose(np.asarray(Image.open(path + image_list[2 * index]))))
        bkd = np.int32(np.transpose(np.asarray(Image.open((path + image_list[2 * index + 1])))))
        # arrays of data, params and fit
        data[index, :, :] = np.absolute(img - bkd)
        fit_data[index, :, :], fit_err[index, :, :] = fitgauss(data[index,:,:])
    else:
        break

# calculate beam diameter, waist (mm) & divergence (deg)
FWHM = 2 * np.sqrt(2 * np.log(2)) * fit_data[:, :, 2] * pix_size
theta_D = np.empty(FWHM.shape)
for col in range(len(FWHM.shape)):
    theta_D[:, col] = np.rad2deg(np.divide(FWHM[:, col], z))

w_0 = wavelen/ (np.pi*theta_D)




# TODO - generate plots of beam fit for each image
