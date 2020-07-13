'''
Sean Keenan, 5th Year MPhys Heriot-Watt University, Edinburgh
Mazerra group
Gaussian Beam Profile Extraction
'''

# import relevant modules
from GFit import *
import os
import numpy as np
import pandas as pd
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as mp
from scipy.optimize import curve_fit

''' Set-up Image file names and processing information '''

# directory name for images
path = '/Users/Message/Desktop/CCD data/40cm lens/'
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
pix_size = chip_size / imsize[0]

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

# calculate beam diameter, waist & divergence (mm)
FWHM = 2 * np.sqrt(2 * np.log(2)) * fit_data[:, :, 2] * pix_size
theta_D = np.empty(FWHM.shape)
for col in range(len(FWHM.shape)):
    theta_D[:, col] = np.divide(FWHM[:, col], z)

# TODO - generate plots of beam fit for each image
