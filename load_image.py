# -*- coding: utf-8 -*-
"""
This is a simple test script 


"""

import numpy as np
import matplotlib.pyplot as plt


def load_raw_as_array(path):
    """
    Reads in .raw image file as a numpy array.

    Parameters
    ----------
    path : str
        Path to the file being loaded.

    Returns
    -------
    im : np.array
        The image as an array of greyscale color intensity values.

    """
    im = np.fromfile(path, dtype='int16', sep="")  # comes in as 1d vector
    im = im.reshape([int(im.shape[0]/(512*512)), 512, 512], order='C')
    return im
    

# TEST
if __name__ == "__main__":
    
    # Set path to images
    directory = './data/subset0-sample-x2'
    
    # Load a CT scan (.raw) into memory using numpy
    filename='1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.raw'
    path = f"{directory}/{filename}"
    ct_scan = load_raw_as_array(path)
    print("ct_scan from .raw:", ct_scan.shape)
    
    # Show an image from the scan
    fig, ax = plt.subplots()
    im = ax.imshow(ct_scan[80], cmap=plt.cm.gray)
