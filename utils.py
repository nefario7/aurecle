import math
from skimage import io, color
from skimage.transform import resize
import numpy as np
from tqdm import tqdm

# A class to initialize the super pixels, of the form - [h,y,l,a,b].
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


# function which returns an object of class SuperPixel
def make_superPixel(h, w, img):
    return SuperPixels(h, w, img[h, w][0], img[h, w][1], img[h, w][2])


def initial_cluster_center(S, img, img_h, img_w, clusters):
    h = S // 2
    w = S // 2
    while h < img_h:
        while w < img_w:
            clusters.append(make_superPixel(h, w, img))
            w += S
        w = S // 2
        h += S
    return clusters


# function to calculate gradient at each pixel
def calc_gradient(h, w, img, img_w, img_h):
    if w + 1 >= img_w:
        w = img_w - 2
    if h + 1 >= img_h:
        h = img_h - 2
    grad = img[w + 1, h + 1][0] - img[w, h][0] + img[w + 1, h + 1][1] - img[w, h][1] + img[w + 1, h + 1][2] - img[w, h][2]
    return grad


# function which reassign the cluster center to the pixel having the lowest gradient
def reassign_cluster_center_acc_to_grad(clusters, img):
    for c in clusters:
        cluster_gradient = calc_gradient(c.h, c.w, img, img_w, img_h)
        for dh in range(-1, 2):
            for dw in range(-1, 2):
                H = c.h + dh
                W = c.w + dw
                new_gradient = calc_gradient(H, W, img, img_w, img_h)
                if new_gradient < cluster_gradient:
                    c.update(H, W, img[H, W][0], img[H, W][1], img[H, W][2])
                    c_gradient = new_gradient


"""
function to assign pixels to the nearest cluster using the Euclidean distance involving both color and spatial
proximity.
"""


def assign_pixels_to_cluster(clusters, S, img, img_h, img_w, tag, dis):
    for c in clusters:
        for h in range(c.h - 2 * S, c.h + 2 * S):
            if h < 0 or h >= img_h:
                continue
            for w in range(c.w - 2 * S, c.w + 2 * S):
                if w < 0 or w >= img_w:
                    continue
                l, a, b = img[h, w]
                Dc = math.sqrt(math.pow(l - c.l, 2) + math.pow(a - c.a, 2) + math.pow(b - c.b, 2))
                Ds = math.sqrt(math.pow(h - c.h, 2) + math.pow(w - c.w, 2))
                D = math.sqrt(math.pow(Dc / m, 2) + math.pow(Ds / S, 2))
                if D < dis[h, w]:
                    if (h, w) not in tag:
                        tag[(h, w)] = c
                        c.pixels.append((h, w))
                    else:
                        tag[(h, w)].pixels.remove((h, w))
                        tag[(h, w)] = c
                        c.pixels.append((h, w))
                    dis[h, w] = D


# function to replace the cluster center with the mean of the pixels contained in the cluster
def update_cluster_mean(clusters):
    for c in clusters:
        sum_h = sum_w = number = 0
        # print("c.pixels",c.pixels)
        for p in c.pixels:
            sum_h += p[0]
            sum_w += p[1]
            number += 1
            H = sum_h // number
            W = sum_w // number
            c.update(H, W, img[H, W][0], img[H, W][1], img[H, W][2])


# replace the color of each pixel in a cluster by the color of the cluster's center
def avg_color_cluster(img, name, clusters):
    image = np.copy(img)
    for c in clusters:
        for p in c.pixels:
            image[p[0], p[1]][0] = c.l
            image[p[0], p[1]][1] = c.a
            image[p[0], p[1]][2] = c.b
        # To change the color of cluster center to Black
        image[c.h, c.w][0] = 0
        image[c.h, c.w][1] = 0
        image[c.h, c.w][2] = 0
    lab2rgb(name, image)


# function to convert LAB images back to RGB and save it
def lab2rgb(path, lab_arr):
    rgb_arr = color.lab2rgb(lab_arr)
    io.imsave(path, rgb_arr)
