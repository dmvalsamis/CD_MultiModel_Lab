import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np

def scaleMinMax(x):
    return ((x - np.nanpercentile(x,2)) / (np.nanpercentile(x,98) - np.nanpercentile(x,2)))

def create_rgb(x):
    r = x[:,:,2]
    r = scaleMinMax(r)
    g = x[:,:,1]
    g = scaleMinMax(g)
    b  = x[:,:,0]
    b = scaleMinMax(b)
    rgb = np.dstack((r,g,b))
    return(rgb)
    
def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1))
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()
    

def image_pair(img1, img2):
	img1 = create_rgb(img1)
	img2 = create_rgb(img2)
	fig, ax = plt.subplots(1, 2, figsize=(10,8), constrained_layout=True)
	ax[0].imshow(img1)
	ax[0].axis('off')
	ax[1].imshow(img2)
	ax[1].axis('off')


def CD (img1, img2, cm):
    # create figure
    fig = plt.figure(figsize=(20, 10), constrained_layout=True)
    #fig.suptitle('true value:',str(y_true),'|predicted value:', str(y_pred), fontsize=5)
    # setting values to rows and column variables
    rows = 1
    columns = 3
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
    # showing image
    plt.imshow(create_rgb(img1))
    plt.axis('off')
    plt.title("pair1")
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    # showing image
    plt.imshow(create_rgb(img2))
    plt.axis('off')
    plt.title("pair2")
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    # showing image
    plt.imshow(cm)
    plt.axis('off')
    plt.title("change_mask")
    


def visualize_features(feature_map):
    fig = plt.figure(figsize=(20, 3))
    rows = 1
    columns = 5
    fig.suptitle('5 first feature maps', fontsize=20)
    fig.add_subplot(rows, columns, 1)
    plt.imshow(feature_map[0, :, :, 0], cmap='gray')
    plt.axis('off')
    fig.add_subplot(rows, columns, 2)
    # showing image
    plt.imshow(feature_map[0, :, :, 1], cmap='gray')
    plt.axis('off')
    fig.add_subplot(rows, columns, 3)
    # showing image
    plt.imshow(feature_map[0, :, :, 2], cmap='gray')
    plt.axis('off')
    fig.add_subplot(rows, columns, 4)
    # showing image
    plt.imshow(feature_map[0, :, :, 3], cmap='gray')
    plt.axis('off')
    fig.add_subplot(rows, columns, 5)
    # showing image
    plt.imshow(feature_map[0, :, :, 4], cmap='gray')
    plt.axis('off')
    
    

    
    
