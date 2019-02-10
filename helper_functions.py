import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
import cv2

#######################################################################################################
# Plotting functions
#######################################################################################################
def show_images(images, figsize=(15, 15), cols=5):
    """Shows images from a dataset. Plots maximal 5 images per row"""
    rows = len(images) // cols
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    cmap = None
    for i, ax in enumerate(axes.flat):
        # check if image is grayscale
        img = images[i]
        if  img.shape[-1] < 3 or len(img.shape) < 3:
            # image is in grayscale
            cmap="gray"
            # image has to be reshaped
            img = img.reshape(img.shape[0], img.shape[1])
        ax.imshow(img, cmap=cmap)
    plt.show()
    return
                

def show_random_images(X, number):
    """Plots a number of random images from a dataset. Plots max 5 images per row"""
    indices = np.random.randint(0, len(X), size=number)
    X = np.array(X)
    X_sel = X[indices]
    show_images(X_sel)
    return

#######################################################################################################
# Image processing functions                             
####################################################################################################### 

def grayscale_images(X):
    """Converts RGB images to grayscale"""
    return np.asarray([grayscale_image(img) for img in X])

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def normalize_grayscale(image_data):
    """
    Normalize the gray scale image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

    
            