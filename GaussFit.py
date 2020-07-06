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
        # convert to list to unpack
        fit_params = params[index, :].tolist()
        fit = gauss_2d(*fit_params)
    else:
        break


''' Test data set - generate dummy gaussian in 2D and then extract FWHM in x and y, plotting fit to data '''

# generate pixel grid
x = np.arange(start=0, stop=960, step=1)
y = np.arange(start=0, stop=1280, step=1)
x, y = np.meshgrid(x, y)

# gaussian attributes
# amplitude
height = np.random.randint(low=5, high=10, size=1)
# centre point x
x_0 = int(np.round(np.max(x)/2))
centre_x = np.random.randint(low=x_0 - x_0 * 0.1, high=x_0 + x_0 * 0.1, size=1)
# centre point y
y_0 = int(np.round(np.max(y)/2))
centre_y = np.random.randint(low=y_0 - y_0 * 0.1, high=y_0 + y_0 * 0.1, size=1)
# width in x
width_x = np.random.randint(low=x_0 - x_0 * 0.8 , high=x_0 - x_0 * 0.6, size=1)
# width in y
width_y = np.random.randint(low=y_0 - y_0 * 0.8, high=y_0 - y_0 * 0.6, size=1)

# generate gaussian and add noise to data
z = gauss_2d(height, centre_x, centre_y, width_x, width_y)(x, y)
noise = 0.5 * np.random.normal(loc=0.6, scale=1, size=z.shape)
z += noise

# plot data
fig, ax = mp.subplots(1, 1)
gauss_test = ax.contourf(x, y, z)
ax.set_title(' Random Beam Profile ')
ax.set(xlabel='Pixel', ylabel='Pixel')
fig.colorbar(gauss_test)

print('finished')