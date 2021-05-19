import cv2
import os

import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from collections import Counter
import seaborn as sns

from PIL import Image


if __name__ == '__main__':

    img = np.array(Image.open('0.png'))
    red = []
    blue = []
    green = []


    for i in range(len(img)):
        for j in range(len(img[0])):
            red.append(img[i][j][0])
            green.append(img[i][j][1])
            blue.append(img[i][j][2])


    sns.distplot(red, hist=False, kde=True)
    sns.distplot(blue, hist=False, kde=True, label="blue")
    sns.distplot(green, hist=False, kde=True, label="green")


    plt.legend()
    plt.show()
