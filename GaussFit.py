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
from scipy import optimize

# fit 2D gaussian to each matrix and extract parameters
# interpolate data to generate 3D model of beam profile
# save and print data to excel sheet

''' Create functions to fit data '''

# define neccesary functions
def gauss_1d(height, centre, width):
    '''Generates Gaussian with given parameters'''
    return lambda x: height * np.exp(-(np.power(x - centre, 2) / (2 * width ** 2)))

# generate 2D gaussian
def gauss_2d(height, centre_x, centre_y, width_x, width_y):
    ''' Generates 2D Gaussian with given parameters:
        height, centre_x, centre_y, width_x, width_y '''
    return lambda x, y : height*np.exp(-(np.power(x - centre_x, 2)/(2 * width_x ** 2) + np.power(y - centre_y, 2) / (2 * width_y ** 2)))

# fit a gaussian to data by calculating its 'moments' (mean, variance, width, height)
def moments(data):
    '''Calculates parameters of a 2D gaussian function by calculating its moments (height, x, y, centre_x, centre_y width_x, width_y'''
    total = data.sum()
    X, Y = np.indices(data.shape)
    centre_x = (X*data).sum()/total
    centre_y = (Y*data).sum()/total
    height = data.max()
    # extract entire column from data of y
    col = data[:, int(centre_y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - centre_x) ** 2 * col).sum() / col.sum())
    row = data[:, int(centre_x)]
    width_y = np.sqrt(np.abs((np.arange(col.size) - centre_y) ** 2 * row).sum() / row.sum())
    return height, centre_x, centre_y, width_x, width_y

def fitgauss_2d(data):
    '''Returns 2D Gaussian parameters from fit (height, x, y, width_x, width_y'''
    params = moments(data)
    err_fun = lambda p: np.ravel(gauss_2d(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.leastsq(err_fun, params)
    return p

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

''' Read image and subtract background data - wont work if uneven number of files in folder '''

# read image and subtract background - store in new array
data = np.empty([int(len(image_list)/2), imsize[0], imsize[1]])
params = np.empty([int(len(image_list)/2), 5])
# fit = np.empty([int(len(image_list)/2), 5])
for index, image in enumerate(image_list):
    # break when half way through data
    if index < int(len(image_list)/2):
        # read then discard image (change to int32 as uint8 does not have -ve values)
        img = np.int32(np.transpose(np.asarray(Image.open(path + image_list[2 * index]))))
        bkd = np.int32(np.transpose(np.asarray(Image.open((path + image_list[2 * index + 1])))))
        # arrays of data, params and fit
        data[index, :, :] = np.absolute(img - bkd)
        params[index, :] = fitgauss_2d(data[index,:,:])
    else:
        break

mp.matshow(data[0,:], cmap=mp.cm.gist_earth_r)

fit = gauss_2d(*params[0,:])

mp.contour(fit(*np.indices(imsize)), cmap=mp.cm.copper)
ax = mp.gca()

print('finished')