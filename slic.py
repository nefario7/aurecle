# SLIC Segmentation
# * Dependencies
import math
import os
import numpy as np
from skimage import io, color
from skimage.transform import resize
from tqdm import tqdm

# * Imports
from utils import *

# * A class to initialize the super pixels, of the form - [h,y,l,a,b].
class SuperPixels(object):
    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []

    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b


# * A class to process images with SLIC Segmentation
class SlicSegmentation:
    base_path = r"sample"

    def __init__(self, m: int, k: int):
        self.clusters = []
        self.tag = {}
        self.k = k
        self.m = m

    # function which returns an object of class SuperPixel
    def __make_superPixel(self, h, w):
        return SuperPixels(h, w, self.img[h, w][0], self.img[h, w][1], self.img[h, w][2])

    # To define the initial cluster centers distanced at S
    def __initial_cluster_center(self, S):
        h = S // 2
        w = S // 2
        while h < self.img_h:
            while w < self.img_w:
                self.clusters.append(self.__make_superPixel(h, w))
                w += S
            w = S // 2
            h += S

    # function to calculate gradient at each pixel
    def __calc_gradient(self, h, w):
        if w + 1 >= self.img_w:
            w = self.img_w - 2
        if h + 1 >= self.img_h:
            h = self.img_h - 2
        grad = (
            self.img[w + 1, h + 1][0]
            - self.img[w, h][0]
            + self.img[w + 1, h + 1][1]
            - self.img[w, h][1]
            + self.img[w + 1, h + 1][2]
            - self.img[w, h][2]
        )
        return grad

    # function which reassign the cluster center to the pixel having the lowest gradient
    def __reassign_cluster_center_acc_to_grad(self):
        for c in self.clusters:
            cluster_gradient = self.__calc_gradient(c.h, c.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    H = c.h + dh
                    W = c.w + dw
                    new_gradient = self.__calc_gradient(H, W)
                    if new_gradient < cluster_gradient:
                        c.update(H, W, self.img[H, W][0], self.img[H, W][1], self.img[H, W][2])
                        c_gradient = new_gradient

    def __assign_pixels_to_cluster(self, S, dis):
        for c in self.clusters:
            for h in range(c.h - 2 * S, c.h + 2 * S):
                if h < 0 or h >= self.img_h:
                    continue
                for w in range(c.w - 2 * S, c.w + 2 * S):
                    if w < 0 or w >= self.img_w:
                        continue
                    l, a, b = self.img[h, w]
                    Dc = math.sqrt(math.pow(l - c.l, 2) + math.pow(a - c.a, 2) + math.pow(b - c.b, 2))
                    Ds = math.sqrt(math.pow(h - c.h, 2) + math.pow(w - c.w, 2))
                    D = math.sqrt(math.pow(Dc / self.m, 2) + math.pow(Ds / S, 2))
                    if D < dis[h, w]:
                        if (h, w) not in self.tag:
                            self.tag[(h, w)] = c
                            c.pixels.append((h, w))
                        else:
                            self.tag[(h, w)].pixels.remove((h, w))
                            self.tag[(h, w)] = c
                            c.pixels.append((h, w))
                        dis[h, w] = D

    # function to replace the cluster center with the mean of the pixels contained in the cluster
    def __update_cluster_mean(self):
        for c in self.clusters:
            sum_h = sum_w = number = 0
            # print("c.pixels",c.pixels)
            for p in c.pixels:
                sum_h += p[0]
                sum_w += p[1]
                number += 1
                H = sum_h // number
                W = sum_w // number
                c.update(H, W, self.img[H, W][0], self.img[H, W][1], self.img[H, W][2])

    # replace the color of each pixel in a cluster by the color of the cluster's center
    def __avg_color_cluster(self, name):
        image = np.copy(self.img)
        for c in self.clusters:
            for p in c.pixels:
                image[p[0], p[1]][0] = c.l
                image[p[0], p[1]][1] = c.a
                image[p[0], p[1]][2] = c.b
            # To change the color of cluster center to Black
            # image[c.h, c.w][0] = 0
            # image[c.h, c.w][1] = 0
            # image[c.h, c.w][2] = 0
        self.__save_output(image, name)
        return image

    def __save_output(self, lab_arr, save_name):
        save_path = os.path.join(self.base_path, save_name)
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(save_path, rgb_arr)

    # function for the Simple Linear Iterative Clustering
    def __slic(self, S, dis, imgname, iterations=10):
        self.__initial_cluster_center(S)
        self.__reassign_cluster_center_acc_to_grad()
        for i in tqdm(range(iterations)):  # usually the algortihm converges within 10 iterations
            self.__assign_pixels_to_cluster(S, dis)
            self.__update_cluster_mean()
            if i == 9:  # to print the output after 10 iterations
                name = "{imgname}_out_m{m}_k{k}.png".format(loop=i, m=self.m, k=self.k, imgname=imgname)
                return_image = self.__avg_color_cluster(name)

        return return_image

    def process(self, image_name):
        image_path = os.path.join(self.base_path, image_name)
        try:
            rgb = io.imread(image_path, plugin="matplotlib")
        except:
            print("File could not be found!")
            exit(0)

        img = resize(rgb, (400, 400), anti_aliasing=True)

        self.img = color.rgb2lab(img)
        self.img_h = img.shape[0]  # Image Height
        self.img_w = img.shape[1]  # Image Width

        N = self.img_h * self.img_w  # Total number of pixels in the image
        S = int(math.sqrt(N / self.k))  # average size of each superpixel

        # initialize the distance between pixels and cluster center as infinity
        dis = np.full((self.img_h, self.img_w), np.inf)
        self.__slic(S, dis, image_name.split(".")[0])
