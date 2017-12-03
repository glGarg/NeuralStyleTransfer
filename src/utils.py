import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats

def create_noise_image(loc = 128.0, scale = 50.0, dim = [224, 224, 3]):
    size = np.prod(dim)
    random_vec = stats.truncnorm.rvs((-loc)/scale, (255-loc)/scale,
                                     loc=loc, scale=scale, size=size)
    gen = np.reshape(random_vec, dim)
    return gen

def open_image(image_path):
    return cv2.imread(image_path)

def normalize_image(image):
    return image - np.array([123.68, 116.779, 103.939])

def show_image(image):
    plt.imshow(np.clip(image + np.array([123.68, 116.779, 103.939]) / 255.0, 0.0, 1.0))
    plt.show()
