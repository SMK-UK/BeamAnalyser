'''
Sean Keenan, 5th Year MPhys Heriot-Watt University, Edinburgh
Mazerra group
Gaussian Beam Profile Extraction
'''

# import relevant modules
import numpy as np
import matplotlib.pyplot as mp
from scipy import optimize

# define neccesary functions
def gauss_1d(height, centre, width):
    '''Generates Gaussian with given parameters'''
    return lambda x: height * np.exp(-(np.power(x - centre, 2) / (2 * width ** 2)))

def gauss_2d(height, centre_x, centre_y, width_x, width_y):
    '''Generates Gaussian with given parameters'''
    return lambda x, y : height*np.exp(-(np.power(x - centre_x, 2)/(2 * width_x ** 2) + np.power(y - centre_y, 2) / (2 * width_y ** 2)))

# can fit a gaussian to data by calculating its 'moments' (mean, variance, width, height)
def moments(data):
    '''Calculates parameters of a 2D gaussian function by calculating its moments (height, x, y, centre_x, centre_y width_x, width_y'''
    total = data.sum()
    X, Y = np.indices(data.shape)
    centre_x = (X*data).sum()/total
    centre_y = (Y*data).sum()/total
    height = data.max()
    # extract entire column from data of y
    col = data[int(centre_x), :]
    width_x = np.sqrt(np.abs((np.arange(col.size) - centre_x) ** 2 * col).sum() / col.sum())
    row = data[:, int(centre_y)]
    width_y = np.sqrt(np.abs((np.arange(row.size) - centre_y) ** 2 * row).sum() / row.sum())
    return height, centre_x, centre_y, width_x, width_y

def fitgauss_2d(data):
    '''Returns 2D Gaussian parameters from fit (height, x, y, width_x, width_y'''
    params = moments(data)
    err_fun = lambda p: np.ravel(gauss_2d(*p)(*np.indices(data.shape)) - data)
    p, success = optimize.least_squares(err_fun, params)
    return p

''' Test data set - generate dummy gaussian in 2D and then extract FWHM in x and y, plotting fit to data '''

# generate pixel grid
x = np.arange(start=0, stop=1280, step=1)
y = np.arange(start=0, stop=960, step=1)
x, y = np.meshgrid(x, y)

# gaussian attributes
# amplitude
height = np.random.randint(low=90, high=120, size=1)
# centre point x
x_0 = int(np.round(np.max(x)/2))
centre_x = np.random.randint(low=550 , high=700, size=1)
# centre point y
y_0 = int(np.round(np.max(y)/2))
centre_y = np.random.randint(low=150, high=200, size=1)
# width in x
width_x = np.random.randint(low=35, high=40, size=1)
# width in y
width_y = np.random.randint(low=35, high=40, size=1)

# generate gaussian and add noise to data
z = gauss_2d(height, centre_x, centre_y, width_x, width_y)(x, y)
noise = 0.5 * np.random.normal(loc=0.6, scale=1, size=z.shape)
z += noise

fig, ax = mp.subplots()
ax.matshow(z, cmap=mp.cm.gist_earth_r)
params = fitgauss_2d(z)
fit = gauss_2d(*params)
ax.contour(fit(*np.indices(z.shape)), cmap=mp.cm.copper)

