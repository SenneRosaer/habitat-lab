import cv2
import os

import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from collections import Counter
import seaborn as sns


if __name__ == '__main__':
    bases = ['../video_dir/2/videos/', '../video_dir/1/videos/']
    f_red = []
    f_blue = []
    f_green = []
    for base in bases:
        files = os.listdir(base)
        images = []
        red = []
        blue = []
        green = []
        for video in files:
            vid = cv2.VideoCapture(base +video)
            prev = None
            while True:
                ret, frame = vid.read()
                if not ret:
                    red.append(prev[:,:256,0:1])
                    green.append(prev[:,:256,1:2])
                    blue.append(prev[:,:256,2:3])

                    break
                prev = frame

        red2 = []
        green2 = []
        blue2 = []

        for index in range(len(red)):
            for i in range (256):
                for j in range(256):
                    t = red[index][i][j]
                    red2.append(t[0])
                    green2.append(green[index][i][j][0])
                    blue2.append(blue[index][i][j][0])

        if f_red == []:
        #     f_red = list((Counter(f_red) - Counter(red2)).elements())
        #     f_green = list((Counter(f_green) - Counter(green2)).elements())
        #     f_blue = list((Counter(f_blue) - Counter(blue2)).elements())
        # else:
            f_red = red2
            f_green = green2
            f_blue = blue2
    # fig, axs = plt.subplots(1,3, tight_layout=True)
    # np.seterr(divide='ignore', invalid='ignore')
    # axs[0].hist(f_red, density=True)
    # axs[0].hist(red2, density=True)
    # axs[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # axs[1].hist(f_green, density=True)
    # axs[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    # axs[2].hist(f_blue, density=True)
    # axs[2].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    #
    # plt.show()

    sns.distplot(f_red, hist=False, kde=True, label="red floor 7")
    sns.distplot(f_blue, hist=False, kde=True, label="blue floor 7")
    sns.distplot(f_green, hist=False, kde=True, label="green floor 7")
    plt.legend()
    plt.show()

    sns.distplot(red2, hist=False, kde=True, label="red floor 6")
    sns.distplot(green2, hist=False, kde=True, label="green floor 6")
    sns.distplot(blue2, hist=False, kde=True, label="blue floor 6")
    plt.legend()
    plt.show()


    #Both together
    sns.distplot(f_red, hist=False, kde=True, label="red floor 7")
    sns.distplot(f_blue, hist=False, kde=True, label="blue floor 7")
    sns.distplot(f_green, hist=False, kde=True, label="green floor 7")

    sns.distplot(red2, hist=False, kde=True, label="red floor 6")
    sns.distplot(green2, hist=False, kde=True, label="green floor 6")
    sns.distplot(blue2, hist=False, kde=True, label="blue floor 6")
    plt.legend()
    plt.show()
