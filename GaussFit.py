'''
Sean Keenan, 5th Year MPhys Heriot-Watt University, Edinburgh
Mazerra group
Gaussian Beam Profile Extraction
'''

# import relevant modules
import numpy as np
import os
from natsort import natsorted
from PIL import Image
import matplotlib.pyplot as mp
from scipy import optimize

# fit 2D gaussian to each matrix and extract parameters
# interpolate data to generate 3D model of beam profile
# save and print data to excel sheet

'''Set-up Image file names and processing information'''

# directory name for images
path = '/Users/Message/Desktop/CCD data/20cm lens/'
file_list = os.listdir(path)
# extract relevant files and sort
image_list = natsorted([i for i in file_list if i.endswith('.bmp')])
# read image and subtract background - store in new array

img = Image.open(path + image_list[0])
imsize = img.size

data = np.empty(imsize, int(len(image_list)/2))
for index, image in enumerate(image_list):
    if index < int(len(image_list)/2):
        img = np.asarray(Image.open(path + image_list[2 * index]))
        bkd = np.asarray(Image.open((path + image_list[2 * index + 1])))
        data = img - bkd
    else:
        break

''' test data set - generate dummy gaussian in 2D and then extract FWHM in x and y, plotting fit to data '''
# generate random gaussian with noise
def gauss_2d(height, centre_x, centre_y, width_x, width_y):
    ''' Generates 2D Gaussian with given parameters:
        height, centre_x, centre_y, width_x, width_y '''
    return lambda x, y : height*np.exp(-(np.power(x - centre_x, 2)/(2 * width_x ** 2) + np.power(y - centre_y, 2) / (2 * width_y ** 2)))

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
