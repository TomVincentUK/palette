import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from colorspacious import cspace_convert
from sklearn.mixture import GaussianMixture

def to_uni(a):
    return cspace_convert(a, 'sRGB1', 'CAM02-UCS')

def from_uni(a):
    return cspace_convert(a, 'CAM02-UCS', 'sRGB1')

def in_gamut_uni(a):
    return np.bitwise_and.reduce(np.abs(a - [50, 0, 0])<50, axis=-1)

def in_gamut_rgb(a):
    return np.bitwise_and.reduce(np.abs(a - 0.5)<0.5, axis=-1)

fig = plt.figure(figsize=(16, 9))
gs = GridSpec(
    figure=fig,
    nrows=2,
    ncols=2
)

# Load and display source image
image = plt.imread('test_image.jpg')
im_ax = fig.add_subplot(gs[0, 0])
im_ax.imshow(image)
im_ax.set(
    title='source image'
)
im_ax.set_axis_off()

# Show it as a pixel wise scatter in perceptually uniform space
max_size = 2**14
columns_rgb = image.reshape(-1, image.shape[-1]) / 255
if columns_rgb.shape[0]>max_size:
    columns_rgb = columns_rgb[
        np.random.choice(columns_rgb.shape[0], max_size, replace=False)
    ]
columns_uni = to_uni(columns_rgb)
scat_ax = fig.add_subplot(gs[0, 1], projection='3d')
scat_ax.scatter(*columns_uni.T, c=columns_rgb, s=1)
scat_ax.set(
    xlabel='$J$',
    ylabel='$a$',
    zlabel='$b$',
    title='pixel scatter'
)

# Perform data clustering on the colour distribution
N_colours = 2
GMM = GaussianMixture(
    n_components=N_colours,
    covariance_type='full'
)
GMM.fit(columns_uni)

# Generate a unit sphere surface
N_surf = 4096
surf = np.random.randn(N_surf, 3)
surf /= np.linalg.norm(surf, axis=-1)[..., np.newaxis]

surf_ax = fig.add_subplot(gs[1, 0], projection='3d')
surf_ax.set(
    xlabel='$J$',
    ylabel='$a$',
    zlabel='$b$',
    title='clusters'
)

N_shades = 5
shades = []
for mean, cov in zip(GMM.means_, GMM.covariances_):
    vals, vecs = np.linalg.eigh(cov)
    ellipse = (vals*vecs @ surf.T).T
    ellipse += mean
    ellipse = ellipse[in_gamut_uni(ellipse)]
    ellipse_rgb = from_uni(ellipse)
    ellipse = ellipse[in_gamut_rgb(ellipse_rgb)]
    ellipse_rgb = ellipse_rgb[in_gamut_rgb(ellipse_rgb)]
    surf_ax.scatter(*ellipse.T, c=ellipse_rgb, s=1)
    
    shades.append(
        np.linspace(mean-(vals*vecs)[:, 0], mean-(vals*vecs)[:, 0], N_shades)
    )
shades = np.array(shades)


fig.tight_layout()
plt.show()