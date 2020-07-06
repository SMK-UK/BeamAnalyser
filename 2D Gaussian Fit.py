'''
Sean Keenan, 5th Year MPhys Heriot-Watt University, Edinburgh
Mazerra group
Gaussian Beam Profile Extraction
'''

# import relevant modules
import numpy as np
import matplotlib.pyplot as mp
from scipy import optimize

# test data set
# generate random gaussian with noise
def gauss_1d(height, centre, width):
    '''Generates Gaussian with given parameters'''
    return lambda x: height * np.exp(-(np.power(x - centre, 2) / (2 * width ** 2)))

def gauss_2d(height, centre_x, centre_y, width_x, width_y):
    '''Generates Gaussian with given parameters'''
    return lambda x, y : height*np.exp(-(np.power(x - centre_x, 2)/(2 * width_x ** 2) + np.power(y - centre_y, 2) / (2 * width_y ** 2)))

# amplitude
height = np.random.randint(low=0, high=20, size=1)
# centre point x
centre_x = np.random.randint(low=1, high=5, size=1)
# centre point y
centre_y = np.random.randint(low=1, high=5, size=1)
# width in x
width_x = np.random.randint(low=50, high=80, size=1)
# width in y
width_y = np.random.randint(low=50, high=80, size=1)

# simulate pixels and generate gaussian
x = np.linspace(start=0, stop=501, num=500)
y = x.copy()
x, y = np.meshgrid(x, y)
z = gauss_2d(height, centre_x, centre_y, width_x, width_y)(x, y)

# generate and add noise to data
noise = 0.5 * np.random.normal(loc=0.5, scale=1, size=z.shape)
z += noise

'''
# plot data
fig_0, ax_0 = mp.subplots(1, 1)
gauss_test = ax.contourf(x, y, z)
ax_0.set_title(' Random Beam Profile ')
ax_0.set(xlabel='Pixel', ylabel='Pixel')
fig_0.colorbar(gauss_test)
mp.show()
'''

mp.matshow(z, cmap=mp.cm.gist_earth_r)

# can fit a gaussian to data by calculating its 'moments' (mean, variance, width, height)
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


params = fitgauss_2d(z)
fit = gauss_2d(*params)

mp.contour(fit(*np.indices(z.shape)), cmap=mp.cm.copper)
ax = mp.gca()

x_params = np.delete(params, [2, 4])
y_params = np.delete(params, [1, 3])