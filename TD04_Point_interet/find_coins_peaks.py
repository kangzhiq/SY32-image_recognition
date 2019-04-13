from skimage import io,util,color,feature
import numpy as np
import skimage
import scipy.ndimage.filters
import matplotlib.pyplot as plt
camera = skimage.data.camera()

camera_f = util.img_as_float(camera)

noyaux_x = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
Ix = scipy.ndimage.filters.convolve(camera_f, noyaux_x, mode='constant')
noyaux_y = np.transpose(noyaux_x)
Iy = scipy.ndimage.filters.convolve(camera_f, noyaux_y, mode='constant')
io.imshow(Ix)

Ix_2 = np.square(Ix)
Iy_2 = np.square(Iy)
Ix_Iy = Ix * Iy

Axx = scipy.ndimage.filters.gaussian_filter(Ix_2, sigma=1, mode='constant')
Ayy = scipy.ndimage.filters.gaussian_filter(Iy_2, sigma=1, mode='constant')
Axy = scipy.ndimage.filters.gaussian_filter(Ix_Iy, sigma=1, mode='constant')
detM = Axx * Ayy - np.square(Axy)
traceM = Axx + Ayy
k = 0.05
R = detM - k * np.square(traceM)

max_res =  scipy.ndimage.filters.maximum_filter(R, size=3)
max_local = R * np.equal(R,max_res)

n_top = 100
max_top = np.isin(max_local,np.sort(np.ravel(max_local))[-1:-(n_top+1):-1])

camera_rgb = color.gray2rgb(camera_f)
def draw_coins(image, max_local, l=3):
    image_rgb = color.gray2rgb(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if max_local[i,j]:
                image_rgb[max(0,i-1):min(image.shape[0],i+l+1), \
                          j, :] = [1, 0, 0]
                image_rgb[i,max(0,j-l):min(image.shape[1],j+l+1), \
                          :] = [1, 0, 0]
    return image_rgb


plt.figure()
plt.clf()
plt.imshow(draw_coins(camera_f, max_top))
